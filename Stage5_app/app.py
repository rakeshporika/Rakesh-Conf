from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
from openai import OpenAI
# For PDF export
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os, io, json
import plotly.express as px
import plotly.graph_objects as go


from load_data import (
    load_versions, load_modules, load_edges, load_metrics,
    load_drift, load_changes_edges, load_changes_modules, load_changes_metrics,
    list_tags, slice_by_tag
)

from config import get_secret
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

from viz_utils import build_graph_html

st.set_page_config(page_title="ArchViz", layout="wide")

# --- EXPLANATION & STATS HELPERS ---------------------------------------------
def legend_block():
    st.markdown(
        """
**Legend**
- 🔵 Blue node/edge: existing module/dependency in the **selected tag**
- 🟢 Green: **added** since the previous tag
- 🔴 Red: **removed** since the previous tag (edges/nodes may not appear if they don't exist in the selected tag)
- Bigger node = higher (fan_in + fan_out)
- Darker blue node = hub (high connectivity)
        """.strip()
    )

def pair_stats(prev_tag: str | None, sel_tag: str | None,
               drift: pd.DataFrame, chg_edges: pd.DataFrame, chg_modules: pd.DataFrame):
    if not prev_tag or not sel_tag:
        st.info("Diff mode off or no previous tag available.")
        return

    row = drift[(drift["tag_from"] == prev_tag) & (drift["tag_to"] == sel_tag)]
    cm = chg_modules[(chg_modules["tag_from"] == prev_tag) & (chg_modules["tag_to"] == sel_tag)]
    ce = chg_edges[(chg_edges["tag_from"] == prev_tag) & (chg_edges["tag_to"] == sel_tag)]

    if row.empty:
        st.info(f"No drift summary for **{prev_tag} → {sel_tag}**.")
        return

    r = row.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules added", int(r["modules_added"]))
    c2.metric("Modules removed", int(r["modules_removed"]))
    c3.metric("Edges added", int(r["edges_added"]))
    c4.metric("Edges removed", int(r["edges_removed"]))

    with st.expander("Show changed modules/edges"):
        a_mods = cm[cm["change_type"] == "module_added"]["module"].head(20).tolist()
        d_mods = cm[cm["change_type"] == "module_removed"]["module"].head(20).tolist()
        st.write("**Added modules (top 20)**", a_mods if a_mods else "—")
        st.write("**Removed modules (top 20)**", d_mods if d_mods else "—")

        a_edges = ce[ce["change_type"] == "edge_added"][["src_module","dst_module"]].head(20)
        d_edges = ce[ce["change_type"] == "edge_removed"][["src_module","dst_module"]].head(20)
        st.write("**Added edges (top 20)**")
        st.dataframe(a_edges, use_container_width=True)
        st.write("**Removed edges (top 20)**")
        st.dataframe(d_edges, use_container_width=True)

def llm_summarize(prompt: str, max_tokens: int = 180) -> str:
    if client is None:
        return "⚠️ LLM key not set. Provide OPENAI_API_KEY to generate insights."
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"Be concise and factual."},
                      {"role":"user","content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM error: {e}"

@st.cache_data(show_spinner=False)
def llm_summarize_cached(prompt: str, max_tokens: int = 500, temperature: float = 0.2):
    if client is None:
        return "⚠️ LLM key not set (OPENAI_API_KEY)."
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"Be precise, evidence-based, and concise."},
                      {"role":"user","content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM error: {e}"

def build_repo_context(repo_slug, versions, drift, metrics):
    total_tags = len(versions) if not versions.empty else 0
    date_min = str(versions["date"].min())[:10] if ("date" in versions.columns and not versions.empty) else "n/a"
    date_max = str(versions["date"].max())[:10] if ("date" in versions.columns and not versions.empty) else "n/a"
    drift_totals = drift[["modules_added","modules_removed","edges_added","edges_removed"]].sum() if not drift.empty else {}
    ctx = {
        "repo": repo_slug,
        "tags": total_tags,
        "period": f"{date_min} → {date_max}",
        "sum_modules_added": int(drift_totals.get("modules_added", 0) or 0),
        "sum_modules_removed": int(drift_totals.get("modules_removed", 0) or 0),
        "sum_edges_added": int(drift_totals.get("edges_added", 0) or 0),
        "sum_edges_removed": int(drift_totals.get("edges_removed", 0) or 0),
        "modules_count_latest": int(metrics[metrics["tag"]==versions["tag"].iloc[-1]].shape[0]) if (not versions.empty and not metrics.empty) else 0
    }
    return ctx

def make_pdf_bytes(title: str, sections: list[tuple[str, str]]) -> bytes:
    """
    sections = [(heading, body_text), ...]
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    hstyle = ParagraphStyle('Heading', parent=styles['Heading2'], spaceAfter=8)
    bstyle = ParagraphStyle('Body', parent=styles['BodyText'], leading=14, spaceAfter=12)

    story = [Paragraph(title, styles['Title']), Spacer(1, 12)]
    for h, body in sections:
        story.append(Paragraph(h, hstyle))
        # Convert line breaks to <br/> for ReportLab Paragraph
        story.append(Paragraph(body.replace("\n", "<br/>"), bstyle))
        story.append(Spacer(1, 6))
    doc.build(story)
    return buf.getvalue()


# ---- Sidebar ----------------------------------------------------------------
st.title("ArchViz: LLM-assisted Architecture Evolution & Visualization")
repo_slug = st.sidebar.text_input("Repo slug", value="fastapi")
base = Path("../data") / repo_slug / "curated"
if not base.exists():
    st.error(f"Data not found at {base}. Please run Stages 2–4.")
    st.stop()


versions = load_versions(repo_slug)
modules = load_modules(repo_slug)
edges = load_edges(repo_slug)
metrics = load_metrics(repo_slug)
drift = load_drift(repo_slug)
chg_edges = load_changes_edges(repo_slug)
chg_modules = load_changes_modules(repo_slug)
chg_metrics = load_changes_metrics(repo_slug)

all_tags = versions["tag"].tolist()


# Default: tag with highest modules_added (fallback to latest tag)
default_idx = len(all_tags) - 1 if all_tags else 0
if not drift.empty:
    max_row = drift.sort_values(["modules_added", "edges_added"], ascending=[False, False]).iloc[0]
    if max_row["tag_to"] in all_tags:
        default_idx = all_tags.index(max_row["tag_to"])

st.sidebar.markdown("### Controls")
sel_tag = st.sidebar.selectbox("Tag", all_tags, index=default_idx if all_tags else 0)

# Diff mode ON by default; auto-derive previous tag
diff_mode = st.sidebar.checkbox("Compare with previous tag", value=True)
idx = all_tags.index(sel_tag) if sel_tag in all_tags else -1
prev_tag = all_tags[idx - 1] if (diff_mode and idx > 0) else None
if prev_tag:
    st.sidebar.markdown(f"Comparing **{prev_tag} → {sel_tag}**")
else:
    st.sidebar.markdown("_No previous tag available for comparison._")

# ---- Tabs -------------------------------------------------------------------
# TAB1, TAB2, TAB3 = st.tabs(["Architecture Graph", "Metrics", "Evolution"])

TAB1, TAB2, TAB3, TAB4 = st.tabs(["Architecture Graph", "Metrics", "Evolution", " LLM Powered Insights"])

# ---- Architecture Graph Tab -------------------------------------------------
with TAB1:
    st.subheader("Per-tag Dependency Graph")

    # Diff highlights if enabled
    added_nodes = removed_nodes = added_edges = removed_edges = None
    if prev_tag:
        cm = chg_modules[(chg_modules["tag_from"] == prev_tag) & (chg_modules["tag_to"] == sel_tag)]
        ce = chg_edges[(chg_edges["tag_from"] == prev_tag) & (chg_edges["tag_to"] == sel_tag)]
        added_nodes = set(cm[cm["change_type"] == "module_added"]["module"].tolist())
        removed_nodes = set(cm[cm["change_type"] == "module_removed"]["module"].tolist())
        added_edges = set(map(tuple, ce[ce["change_type"] == "edge_added"][["src_module","dst_module"]].to_records(index=False)))
        removed_edges = set(map(tuple, ce[ce["change_type"] == "edge_removed"][["src_module","dst_module"]].to_records(index=False)))

    html = build_graph_html(
        sel_tag, modules, edges, metrics,
        added_nodes=added_nodes, removed_nodes=removed_nodes,
        added_edges=added_edges, removed_edges=removed_edges,
        height="720px", width="100%"
    )
    st.components.v1.html(html, height=760, scrolling=True)

    st.markdown(f"**Selected tag:** `{sel_tag}`" + (f"  •  **Compared with:** `{prev_tag}`" if prev_tag else ""))

    legend_block()
    pair_stats(prev_tag, sel_tag, drift, chg_edges, chg_modules)

    with st.expander("How to read this graph"):
        st.write(
            "- The large center indicates hub modules (many imports).\n"
            "- Long spokes are leaf modules that depend on the core.\n"
            "- Green items = newly added in the selected tag vs previous.\n"
            "- Red items can be absent if they no longer exist in the current tag."
        )


# ---- Metrics Tab ------------------------------------------------------------
with TAB2:
    st.subheader("Module Metrics")
    st.caption(
        "Fan-in: #modules that import this module • "
        "Fan-out: #modules this module imports • "
        "Cyclomatic: decision complexity (higher = harder to test)"
    )

    mx = metrics[metrics["tag"] == sel_tag].copy()

    # Top fan-in / fan-out / cyclomatic
    col1, col2, col3 = st.columns(3)
    top_fi = mx.sort_values("fan_in", ascending=False).head(10)[["module","fan_in"]]
    top_fo = mx.sort_values("fan_out", ascending=False).head(10)[["module","fan_out"]]
    top_cy = mx.sort_values("cyclomatic", ascending=False).head(10)[["module","cyclomatic"]]

    with col1:
        st.write("Top fan-in")
        st.dataframe(top_fi, use_container_width=True)
    with col2:
        st.write("Top fan-out")
        st.dataframe(top_fo, use_container_width=True)
    with col3:
        st.write("Top cyclomatic")
        st.dataframe(top_cy, use_container_width=True)

    # Optional charts
    colA, colB = st.columns(2)
    figA = px.histogram(mx, x="fan_in", nbins=30, title="Distribution: fan_in")
    figB = px.histogram(mx, x="cyclomatic", nbins=30, title="Distribution: cyclomatic")
    colA.plotly_chart(figA, use_container_width=True)
    colB.plotly_chart(figB, use_container_width=True)
    with st.expander("How to read these metrics"):
        st.write(
            "- **High fan-in** modules are reusable but risky to change (many depend on them).\n"
            "- **High fan-out** modules are tightly coupled to others (fragile to upstream changes).\n"
            "- **High cyclomatic** modules may need refactoring or extra tests."
        )


# ---- Evolution Tab ----------------------------------------------------------
# with TAB3:
#     st.subheader("Evolution — Tag-to-Tag Drift")
#     if drift.empty:
#         st.info("No drift_summary.csv found.")
#     else:
#         # Simple overview chart across tag pairs
#         df_plot = drift.copy()
#         df_plot["pair"] = df_plot["tag_from"].astype(str) + " → " + df_plot["tag_to"].astype(str)
#         fig = px.bar(df_plot, x="pair", y=["modules_added","modules_removed","edges_added","edges_removed"],
#                      barmode="group", title="Changes per tag pair")
#         st.plotly_chart(fig, use_container_width=True)

#         st.write("Raw drift summary")
#         st.dataframe(df_plot, use_container_width=True)

#         if prev_tag:
#             st.markdown("**Metric deltas for selected pair**")
#             cmx = chg_metrics[(chg_metrics["tag_from"] == prev_tag) & (chg_metrics["tag_to"] == sel_tag)].copy()
#             if not cmx.empty:
#                 # show top movers by |fan_in_delta| and |cyclomatic_delta|
#                 left, right = st.columns(2)
#                 top_fi = cmx.assign(abs_fi=cmx["fan_in_delta"].abs()).sort_values("abs_fi", ascending=False).head(10)
#                 top_cy = cmx.assign(abs_c=cmx["cyclomatic_delta"].abs()).sort_values("abs_c", ascending=False).head(10)
#                 with left:
#                     st.write("Top |fan_in_delta|")
#                     st.dataframe(top_fi[["module","fan_in_delta","fan_out_delta","cyclomatic_delta"]], use_container_width=True)
#                 with right:
#                     st.write("Top |cyclomatic_delta|")
#                     st.dataframe(top_cy[["module","cyclomatic_delta","fan_in_delta","fan_out_delta"]], use_container_width=True)
#             else:
#                 st.info("No metric deltas for this pair.")


# ---- Evolution Tab ----------------------------------------------------------
with TAB3:
    st.subheader("Evolution — Tag-to-Tag Drift")

    if drift.empty:
        st.info("No drift_summary.csv found.")
    else:
        st.markdown(
            "This chart shows **how many modules/edges were added or removed** between each release pair."
        )

        # Overview chart across tag pairs
        df_plot = drift.copy()
        df_plot["pair"] = df_plot["tag_from"].astype(str) + " → " + df_plot["tag_to"].astype(str)
        fig = px.bar(
            df_plot, x="pair",
            y=["modules_added", "modules_removed", "edges_added", "edges_removed"],
            barmode="group", title="Changes per tag pair"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Selected pair details (if diff mode is on and a previous tag exists)
        if prev_tag:
            st.markdown(f"**Selected pair:** `{prev_tag}` → `{sel_tag}`")

            pair_row = drift[(drift["tag_from"] == prev_tag) & (drift["tag_to"] == sel_tag)]
            cm = chg_modules[(chg_modules["tag_from"] == prev_tag) & (chg_modules["tag_to"] == sel_tag)]
            ce = chg_edges[(chg_edges["tag_from"] == prev_tag) & (chg_edges["tag_to"] == sel_tag)]

            if not pair_row.empty:
                r = pair_row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Modules added", int(r["modules_added"]))
                c2.metric("Modules removed", int(r["modules_removed"]))
                c3.metric("Edges added", int(r["edges_added"]))
                c4.metric("Edges removed", int(r["edges_removed"]))

                with st.expander("Show changed modules/edges"):
                    added_mods = cm.query("change_type == 'module_added'")["module"].head(20).tolist()
                    removed_mods = cm.query("change_type == 'module_removed'")["module"].head(20).tolist()

                    st.write("**Added modules (top 20)**", added_mods if added_mods else "—")
                    st.write("**Removed modules (top 20)**", removed_mods if removed_mods else "—")

                    st.write("**Added edges (top 20)**")
                    st.dataframe(
                        ce.query("change_type == 'edge_added'")[["src_module", "dst_module"]].head(20),
                        use_container_width=True
                    )
                    st.write("**Removed edges (top 20)**")
                    st.dataframe(
                        ce.query("change_type == 'edge_removed'")[["src_module", "dst_module"]].head(20),
                        use_container_width=True
                    )

                st.caption(
                    "Tip: Select the **right-hand tag** of this pair in the sidebar with "
                    "“Compare with previous tag” enabled to see **added nodes/edges in green** on the graph."
                )

                # Metric deltas for the selected pair
                st.markdown("**Metric deltas for selected pair**")
                cmx = chg_metrics[(chg_metrics["tag_from"] == prev_tag) & (chg_metrics["tag_to"] == sel_tag)].copy()
                if not cmx.empty:
                    left, right = st.columns(2)
                    top_fi = cmx.assign(abs_fi=cmx["fan_in_delta"].abs()).sort_values("abs_fi", ascending=False).head(10)
                    top_cy = cmx.assign(abs_c=cmx["cyclomatic_delta"].abs()).sort_values("abs_c", ascending=False).head(10)
                    with left:
                        st.write("Top |fan_in_delta|")
                        st.dataframe(
                            top_fi[["module", "fan_in_delta", "fan_out_delta", "cyclomatic_delta"]],
                            use_container_width=True
                        )
                    with right:
                        st.write("Top |cyclomatic_delta|")
                        st.dataframe(
                            top_cy[["module", "cyclomatic_delta", "fan_in_delta", "fan_out_delta"]],
                            use_container_width=True
                        )
                else:
                    st.info("No metric deltas for this pair.")
            else:
                st.info("No drift summary for this selected pair.")

        # Raw table at the end
        st.write("Raw drift summary")
        st.dataframe(df_plot, use_container_width=True)


# ---- Insights Tab -----------------------------------------------------------
with TAB4:
    st.subheader("LLM Powered Insights")

    # Build factual context to ground the LLM
    context = build_repo_context(repo_slug, versions, drift, metrics)

    # ----------------------- (1) Non-Technical Summary -----------------------
    st.markdown("### Non-technical overview")
    nt_prompt = (
        "You are writing for a non-technical audience. In 6–7 lines, neutrally summarize:\n"
        f"- What the repository '{context['repo']}' is and broadly what it does.\n"
        f"- Timeframe analyzed: {context['period']}, number of tags: {context['tags']}.\n"
        f"- High-level architectural character (e.g., hub-and-spoke, layered).\n"
        "- What a 'dependency graph' represents in simple terms.\n"
        "- What 'added/removed modules and edges' mean at release boundaries.\n"
        f"- Emphasize that observations are based on structural metrics and version diffs; avoid speculation.\n"
    )
    nt_text = llm_summarize_cached(nt_prompt, max_tokens=420, temperature=0.2)
    st.write(nt_text)

    # ---- Non-technical visuals ---------------------------------------------------
    lastN = 25  # limit to keep charts readable

    # A) Size over time (modules & edges per tag)
    if not versions.empty:
        tag_order = versions["tag"].tolist()[-lastN:]
        mod_counts = modules.groupby("tag").size().reindex(tag_order, fill_value=0).reset_index(name="modules")
        edge_counts = edges.groupby("tag").size().reindex(tag_order, fill_value=0).reset_index(name="edges")
        size_df = mod_counts.merge(edge_counts, on="tag", how="outer").fillna(0)

        fig_size = px.line(size_df, x="tag", y=["modules","edges"], markers=True,
                        title="Project size over time (modules & edges)")
        fig_size.update_layout(xaxis_title="Tag", yaxis_title="Count")
        st.plotly_chart(fig_size, use_container_width=True)

    # B) Release impact mini-bars (last 8 pairs)
    if not drift.empty:
        mini = drift.tail(8).copy()
        mini["pair"] = mini["tag_from"].astype(str) + " → " + mini["tag_to"].astype(str)

        c1, c2 = st.columns(2)
        with c1:
            fig_m = px.bar(mini, x="pair", y=["modules_added","modules_removed"],
                        barmode="group", title="Modules: added vs removed (last 8 pairs)")
            st.plotly_chart(fig_m, use_container_width=True)
        with c2:
            fig_e = px.bar(mini, x="pair", y=["edges_added","edges_removed"],
                        barmode="group", title="Edges: added vs removed (last 8 pairs)")
            st.plotly_chart(fig_e, use_container_width=True)

    st.divider()

    # -------------------------- (2) Technical Summary ------------------------
    st.markdown("### Technical architecture summary")
    # Provide compact metrics snapshot to the LLM
    latest_tag = versions["tag"].iloc[-1] if not versions.empty else None
    latest_mx = metrics[metrics["tag"] == latest_tag] if latest_tag else pd.DataFrame()
    # crude aggregates (safe if DataFrame empty)
    fi_mean = float(latest_mx["fan_in"].mean()) if not latest_mx.empty else 0.0
    fo_mean = float(latest_mx["fan_out"].mean()) if not latest_mx.empty else 0.0
    cy_p95  = float(latest_mx["cyclomatic"].quantile(0.95)) if not latest_mx.empty else 0.0

    tech_prompt = (
        "Write a 6–7 line technical summary for experienced developers, using only the provided evidence.\n"
        f"EVIDENCE:\n"
        f"- Latest tag: {latest_tag}\n"
        f"- Total modules (latest): {context['modules_count_latest']}\n"
        f"- Mean fan-in (latest): {fi_mean:.2f}, mean fan-out (latest): {fo_mean:.2f}\n"
        f"- 95th percentile cyclomatic (latest): {cy_p95:.0f}\n"
        f"- Cumulative adds/removes (all pairs): modules +{context['sum_modules_added']} / -{context['sum_modules_removed']}, "
        f"edges +{context['sum_edges_added']} / -{context['sum_edges_removed']}\n"
        "- Explain likely architectural shape (e.g., hub modules, peripheral leaves) and potential maintenance implications.\n"
        "- Avoid guessing internal business logic; stick to structure and metrics."
    )
    tech_text = llm_summarize_cached(tech_prompt, max_tokens=480, temperature=0.2)
    st.write(tech_text)

    # ---- Technical visuals -------------------------------------------------------
    if latest_tag:
        mx = metrics[metrics["tag"] == latest_tag].copy()

        if not mx.empty:
            # C) Top fan-in (latest)
            top_fi = mx.nlargest(10, "fan_in")[["module","fan_in"]]
            fig_fi = px.bar(top_fi, x="fan_in", y="module", orientation="h",
                            title=f"Top fan-in modules @ {latest_tag}")
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_fi, use_container_width=True)

            # D) Coupling vs Complexity scatter
            mx_scatter = mx.copy()
            mx_scatter["size"] = mx_scatter["cyclomatic"].clip(lower=1)
            fig_sc = px.scatter(mx_scatter, x="fan_in", y="fan_out",
                                size="size", hover_name="module",
                                title=f"Coupling vs Complexity @ {latest_tag}",
                                labels={"fan_in":"Fan-in","fan_out":"Fan-out","size":"Cyclomatic"})
            st.plotly_chart(fig_sc, use_container_width=True)

            # E) Namespace treemap (prefix by first 2 segments)
            def prefix2(name: str) -> str:
                parts = name.split(".")
                return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            treemap_df = mx.assign(prefix=mx["module"].map(prefix2))
            agg = treemap_df.groupby("prefix", as_index=False).agg(
                modules=("module","count"),
                fan_in_sum=("fan_in","sum")
            )
            fig_tm = px.treemap(agg, path=["prefix"], values="fan_in_sum",
                                color="modules", title="Namespace treemap (size=Σfan_in, color=#modules)")
            st.plotly_chart(fig_tm, use_container_width=True)

    st.divider()

    # ---------------------- (3) Pair-specific Evolution ----------------------
    st.markdown("### Evolution insights for selected pair")

    if prev_tag:
        cm = chg_modules[(chg_modules["tag_from"] == prev_tag) & (chg_modules["tag_to"] == sel_tag)]
        ce = chg_edges[(chg_edges["tag_from"] == prev_tag) & (chg_edges["tag_to"] == sel_tag)]
        cmx = chg_metrics[(chg_metrics["tag_from"] == prev_tag) & (chg_metrics["tag_to"] == sel_tag)]

        added_mods = cm.loc[cm["change_type"]=="module_added","module"].head(25).tolist()
        removed_mods = cm.loc[cm["change_type"]=="module_removed","module"].head(25).tolist()
        added_edges = ce.loc[ce["change_type"]=="edge_added",["src_module","dst_module"]].head(40).values.tolist()
        removed_edges = ce.loc[ce["change_type"]=="edge_removed",["src_module","dst_module"]].head(40).values.tolist()

        # Top movers by absolute delta
        top_fi = (cmx.assign(abs_fi=cmx["fan_in_delta"].abs())
                    .sort_values("abs_fi", ascending=False)
                    .head(10)[["module","fan_in_delta","fan_out_delta","cyclomatic_delta"]].to_dict(orient="records"))
        top_cy = (cmx.assign(abs_c=cmx["cyclomatic_delta"].abs())
                    .sort_values("abs_c", ascending=False)
                    .head(10)[["module","cyclomatic_delta","fan_in_delta","fan_out_delta"]].to_dict(orient="records"))

        ev_prompt = (
            f"Summarize the architectural evolution between {prev_tag} → {sel_tag}.\n"
            "Write 6–7 lines followed by 3–5 crisp bullets. Use only the facts below.\n"
            "Mention concrete module names where helpful; avoid speculation.\n\n"
            f"EVIDENCE:\n"
            f"- Added modules (sample): {added_mods}\n"
            f"- Removed modules (sample): {removed_mods}\n"
            f"- Added edges (sample): {added_edges[:10]}\n"
            f"- Removed edges (sample): {removed_edges[:10]}\n"
            f"- Top |fan_in_delta| modules: {top_fi}\n"
            f"- Top |cyclomatic_delta| modules: {top_cy}\n"
        )
        ev_text = llm_summarize_cached(ev_prompt, max_tokens=520, temperature=0.2)
        st.write(ev_text)
        # ---- Pair visuals ------------------------------------------------------------
        # F) Pair churn bars
        pair_row = drift[(drift["tag_from"] == prev_tag) & (drift["tag_to"] == sel_tag)]
        if not pair_row.empty:
            r = pair_row.iloc[0]
            churn_df = pd.DataFrame({
                "type": ["modules_added","modules_removed","edges_added","edges_removed"],
                "count": [int(r["modules_added"]), int(r["modules_removed"]),
                        int(r["edges_added"]), int(r["edges_removed"])]
            })
            fig_churn = px.bar(churn_df, x="type", y="count", title=f"Release impact: {prev_tag} → {sel_tag}")
            st.plotly_chart(fig_churn, use_container_width=True)

        # G) Top |delta| bars
        cmx_pair = chg_metrics[(chg_metrics["tag_from"] == prev_tag) & (chg_metrics["tag_to"] == sel_tag)].copy()
        if not cmx_pair.empty:
            colA, colB = st.columns(2)

            with colA:
                tf = (cmx_pair.assign(abs_fi=cmx_pair["fan_in_delta"].abs())
                                .nlargest(10, "abs_fi")[["module","fan_in_delta"]])
                fig_tf = px.bar(tf, x="fan_in_delta", y="module", orientation="h",
                                title="Top |fan_in_delta|", color="fan_in_delta",
                                color_continuous_scale="Blues")
                fig_tf.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_tf, use_container_width=True)

            with colB:
                tc = (cmx_pair.assign(abs_c=cmx_pair["cyclomatic_delta"].abs())
                                .nlargest(10, "abs_c")[["module","cyclomatic_delta"]])
                fig_tc = px.bar(tc, x="cyclomatic_delta", y="module", orientation="h",
                                title="Top |cyclomatic_delta|", color="cyclomatic_delta",
                                color_continuous_scale="Purples")
                fig_tc.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_tc, use_container_width=True)

        # H) Optional Sankey for added edges (toggle)
        show_sankey = st.checkbox("Show Sankey for added edges (cap 40 links)", value=False)
        if show_sankey:
            add_edges = (chg_edges[(chg_edges["tag_from"] == prev_tag) & (chg_edges["tag_to"] == sel_tag) &
                                (chg_edges["change_type"] == "edge_added")]
                        [["src_module","dst_module"]].head(40))
            if not add_edges.empty:
                nodes = pd.Index(sorted(set(add_edges["src_module"]) | set(add_edges["dst_module"]))).tolist()
                idx = {n:i for i,n in enumerate(nodes)}
                link = dict(
                    source=[idx[s] for s in add_edges["src_module"]],
                    target=[idx[d] for d in add_edges["dst_module"]],
                    value=[1]*len(add_edges)
                )
                fig_sk = go.Figure(data=[go.Sankey(
                    node=dict(label=nodes, pad=10, thickness=12),
                    link=link
                )])
                fig_sk.update_layout(title_text=f"Added dependencies: {prev_tag} → {sel_tag}", font_size=10)
                st.plotly_chart(fig_sk, use_container_width=True)
            else:
                st.info("No added edges for this pair.")

    else:
        st.info("Select a tag with a previous version (diff mode ON) to generate pair-specific insights.")

    st.divider()

    # ---------------------- (Optional) Comparative & Risk --------------------
    st.markdown("### Comparative context & maintainability ")
    comp_prompt = (
        "Based on the provided structural indicators, compare this repository's likely modularity and coupling\n"
        "against typical Python web frameworks (e.g., Django/Flask). Stay general, avoid brand-new facts:\n"
        f"- Total tags: {context['tags']}, overall adds/removes: +M{context['sum_modules_added']}/-M{context['sum_modules_removed']}, "
        f"+E{context['sum_edges_added']}/-E{context['sum_edges_removed']}.\n"
        "Explain where this project probably sits on a spectrum from monolithic to modular, and provide a short\n"
        "maintainability verdict (Low/Medium/High) with justification grounded in the metrics.\n"
        "6–7 lines, neutral tone."
    )
    comp_text = llm_summarize_cached(comp_prompt, max_tokens=460, temperature=0.2)
    st.write(comp_text)

    # ---- Comparative visuals -----------------------------------------------------
    if not metrics.empty:
        # I) Concentration index: Top-5 fan_in share over tags
        tags_for_ci = versions["tag"].tolist()[-lastN:]
        rows = []
        for t in tags_for_ci:
            mxt = metrics[metrics["tag"] == t]
            if mxt.empty: 
                continue
            total_fi = mxt["fan_in"].sum()
            top5 = mxt.nlargest(5, "fan_in")["fan_in"].sum()
            share = (top5 / total_fi) if total_fi > 0 else 0.0
            rows.append({"tag": t, "top5_fan_in_share": share})
        ci_df = pd.DataFrame(rows)
        if not ci_df.empty:
            fig_ci = px.line(ci_df, x="tag", y="top5_fan_in_share",
                            title="Centralization over time (Top-5 fan-in share)")
            fig_ci.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_ci, use_container_width=True)

        # J) Cyclomatic distribution trend (box by tag)
        mx_last = metrics[metrics["tag"].isin(tags_for_ci)]
        if not mx_last.empty:
            fig_box = px.box(mx_last, x="tag", y="cyclomatic",
                            title="Cyclomatic complexity distribution (last tags)")
            st.plotly_chart(fig_box, use_container_width=True)


    # -------------------------- Export to PDF Button --------------------------
    st.markdown("### Export")
    pdf_sections = [
        ("Non-technical overview", nt_text),
        ("Technical architecture summary", tech_text),
        (f"Evolution insights ({prev_tag} → {sel_tag})" if prev_tag else "Evolution insights", ev_text if prev_tag else "Select a comparable pair to generate details."),
        ("Comparative context & maintainability", comp_text),
    ]
    pdf_bytes = make_pdf_bytes(f"AI Insights — {repo_slug}", pdf_sections)
    st.download_button(
        label="📄 Export insights as PDF",
        data=pdf_bytes,
        file_name=f"insights_{repo_slug}.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ---- Footer -----------------------------------------------------------------
st.caption("Developed by IIT Gandhinagar - ArchViz • Data from Open-Source • PyVis + Plotly • Streamlit")