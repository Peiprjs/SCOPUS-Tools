"""
prism_exporter.py — GraphPad Prism (.pzfx) Project Generator.

Builds standardized GraphPad Prism XML project files (PrismXMLVersion 5.00)
containing multi-table and multi-graph datasets across all executed batch jobs.
Integrates comparative sectoral affiliations, geopolitical distributions,
annual publication dynamics, bibliometric indicators, top countries,
institutions, and target keywords into a single exportable .pzfx file.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import collections
import html
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batch_dataset_exporter import JobResult

logger = logging.getLogger(__name__)

PRISM_XML_NAMESPACE = "http://graphpad.com/prism/Prism.htm"


def _escape(text: Any) -> str:
    """Escape XML special characters in string values."""
    if text is None:
        return ""
    return html.escape(str(text).strip(), quote=True)


def generate_prism_xml(
    results: list[JobResult],
    project_title: str = "Scopus Bibliometric Mass Export",
) -> str:
    """Generate a valid GraphPad Prism XML project (.pzfx) with multiple comparative tables.

    Parameters
    ----------
    results:
        List of completed JobResult instances from the batch processing engine.
    project_title:
        Title recorded in the Prism project Info sheet.

    Returns
    -------
    UTF-8 encoded XML string conforming to GraphPad Prism 5.00 specification.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Filter out empty results if needed, or keep for complete representation
    valid_results = [r for r in results if not r.df.empty]
    cohort_results = valid_results if valid_results else results

    table_ids: list[str] = []
    tables_xml: list[str] = []
    table_counter = 0

    # -----------------------------------------------------------------------
    # Table 1: Sectoral Affiliation Dynamics (OneWay / Column Graph)
    # -----------------------------------------------------------------------
    t_id = f"Table{table_counter}"
    table_ids.append(t_id)
    table_counter += 1

    sector_rows = ["Academia", "Industry", "Mixed", "Unknown"]
    t1_lines = [
        f'<Table ID="{t_id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
        '  <Title>Sectoral Affiliation Dynamics</Title>',
        '  <RowTitlesColumn Width="140">',
        '    <Subcolumn>',
    ]
    for row_name in sector_rows:
        t1_lines.append(f'      <d>{_escape(row_name)}</d>')
    t1_lines.extend([
        '    </Subcolumn>',
        '  </RowTitlesColumn>',
    ])

    for r in cohort_results:
        job_name = _escape(r.job_spec.job_name)
        cat_counts = r.df["category"].value_counts().to_dict() if not r.df.empty and "category" in r.df.columns else {}
        t1_lines.extend([
            f'  <YColumn Width="130" Decimals="0" Subcolumns="1">',
            f'    <Title>{job_name}</Title>',
            '    <Subcolumn>',
        ])
        for row_name in sector_rows:
            cnt = int(cat_counts.get(row_name, 0))
            t1_lines.append(f'      <d>{cnt}</d>')
        t1_lines.extend([
            '    </Subcolumn>',
            '  </YColumn>',
        ])
    t1_lines.append('</Table>')
    tables_xml.append("\n".join(t1_lines))

    # -----------------------------------------------------------------------
    # Table 2: Geopolitical Classification (OneWay / Column Graph)
    # -----------------------------------------------------------------------
    t_id = f"Table{table_counter}"
    table_ids.append(t_id)
    table_counter += 1

    geo_rows = ["EU/EEC", "Non-EU/EEC", "Mixed Geo", "Unknown Geo"]
    t2_lines = [
        f'<Table ID="{t_id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
        '  <Title>Geopolitical Classification</Title>',
        '  <RowTitlesColumn Width="140">',
        '    <Subcolumn>',
    ]
    for row_name in geo_rows:
        t2_lines.append(f'      <d>{_escape(row_name)}</d>')
    t2_lines.extend([
        '    </Subcolumn>',
        '  </RowTitlesColumn>',
    ])

    for r in cohort_results:
        job_name = _escape(r.job_spec.job_name)
        geo_counts = r.df["geo_category"].value_counts().to_dict() if not r.df.empty and "geo_category" in r.df.columns else {}
        t2_lines.extend([
            f'  <YColumn Width="130" Decimals="0" Subcolumns="1">',
            f'    <Title>{job_name}</Title>',
            '    <Subcolumn>',
        ])
        for row_name in geo_rows:
            cnt = int(geo_counts.get(row_name, 0))
            t2_lines.append(f'      <d>{cnt}</d>')
        t2_lines.extend([
            '    </Subcolumn>',
            '  </YColumn>',
        ])
    t2_lines.append('</Table>')
    tables_xml.append("\n".join(t2_lines))

    # -----------------------------------------------------------------------
    # Table 3: Annual Publication Dynamics (XY Graph)
    # -----------------------------------------------------------------------
    all_years_set: set[int] = set()
    for r in cohort_results:
        if not r.df.empty and "year" in r.df.columns:
            for y in r.df["year"].dropna():
                try:
                    iy = int(y)
                    if 1900 <= iy <= 2100:
                        all_years_set.add(iy)
                except (ValueError, TypeError):
                    pass

    sorted_years = sorted(list(all_years_set))
    if not sorted_years:
        current_year = datetime.now().year
        sorted_years = list(range(current_year - 5, current_year + 1))

    t_id = f"Table{table_counter}"
    table_ids.append(t_id)
    table_counter += 1

    t3_lines = [
        f'<Table ID="{t_id}" XFormat="numbers" YFormat="replicates" Replicates="1" TableType="XY" EVFormat="AsteriskAfterNumber">',
        '  <Title>Annual Publication Dynamics</Title>',
        '  <XColumn Width="100" Subcolumns="1" Decimals="0">',
        '    <Title>Publication Year</Title>',
        '    <Subcolumn>',
    ]
    for y in sorted_years:
        t3_lines.append(f'      <d>{y}</d>')
    t3_lines.extend([
        '    </Subcolumn>',
        '  </XColumn>',
    ])

    for r in cohort_results:
        job_name = _escape(r.job_spec.job_name)
        yr_counts: dict[int, int] = {}
        if not r.df.empty and "year" in r.df.columns:
            for y in r.df["year"].dropna():
                try:
                    iy = int(y)
                    yr_counts[iy] = yr_counts.get(iy, 0) + 1
                except (ValueError, TypeError):
                    pass

        t3_lines.extend([
            f'  <YColumn Width="130" Decimals="0" Subcolumns="1">',
            f'    <Title>{job_name}</Title>',
            '    <Subcolumn>',
        ])
        for y in sorted_years:
            t3_lines.append(f'      <d>{yr_counts.get(y, 0)}</d>')
        t3_lines.extend([
            '    </Subcolumn>',
            '  </YColumn>',
        ])
    t3_lines.append('</Table>')
    tables_xml.append("\n".join(t3_lines))

    # -----------------------------------------------------------------------
    # Table 4: Key Bibliometric Indicators (OneWay / Column Graph)
    # -----------------------------------------------------------------------
    t_id = f"Table{table_counter}"
    table_ids.append(t_id)
    table_counter += 1

    kpi_rows = [
        "Total Papers Analyzed",
        "EU/EEC Engagement Rate (%)",
        "Keyword Match Rate (%)",
        "Full-Text Retrieval Rate (%)",
        "Abstract Fallback Rate (%)",
    ]

    t4_lines = [
        f'<Table ID="{t_id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
        '  <Title>Comparative Bibliometric Indicators</Title>',
        '  <RowTitlesColumn Width="220">',
        '    <Subcolumn>',
    ]
    for r_name in kpi_rows:
        t4_lines.append(f'      <d>{_escape(r_name)}</d>')
    t4_lines.extend([
        '    </Subcolumn>',
        '  </RowTitlesColumn>',
    ])

    for r in cohort_results:
        job_name = _escape(r.job_spec.job_name)
        total_p = len(r.df)
        if total_p > 0:
            eu_cnt = int(((r.df.get("geo_category") == "EU/EEC") | (r.df.get("geo_category") == "Mixed Geo")).sum())
            eu_rate = round(eu_cnt / total_p * 100, 2)
            kw_cnt = int(r.df["has_any_keyword"].sum()) if "has_any_keyword" in r.df.columns else 0
            kw_rate = round(kw_cnt / total_p * 100, 2)
            ft_cnt = int((r.df.get("text_source") == "Full Text").sum())
            ft_rate = round(ft_cnt / total_p * 100, 2)
            abs_cnt = int((r.df.get("text_source") == "Abstract").sum())
            abs_rate = round(abs_cnt / total_p * 100, 2)
        else:
            eu_rate = kw_rate = ft_rate = abs_rate = 0.0

        t4_lines.extend([
            f'  <YColumn Width="130" Decimals="2" Subcolumns="1">',
            f'    <Title>{job_name}</Title>',
            '    <Subcolumn>',
            f'      <d>{total_p}</d>',
            f'      <d>{eu_rate}</d>',
            f'      <d>{kw_rate}</d>',
            f'      <d>{ft_rate}</d>',
            f'      <d>{abs_rate}</d>',
            '    </Subcolumn>',
            '  </YColumn>',
        ])
    t4_lines.append('</Table>')
    tables_xml.append("\n".join(t4_lines))

    # -----------------------------------------------------------------------
    # Table 5: Top Publishing Countries (OneWay / Column Graph)
    # -----------------------------------------------------------------------
    country_totals: dict[str, int] = collections.defaultdict(int)
    for r in cohort_results:
        if not r.df.empty and "countries" in r.df.columns:
            for c_list in r.df["countries"]:
                if isinstance(c_list, list):
                    for c in set(c_list):
                        if c and str(c).strip():
                            country_totals[str(c).strip()] += 1

    top_countries = [c for c, _ in collections.Counter(country_totals).most_common(15)]
    if top_countries:
        t_id = f"Table{table_counter}"
        table_ids.append(t_id)
        table_counter += 1

        t5_lines = [
            f'<Table ID="{t_id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
            '  <Title>Top Publishing Countries</Title>',
            '  <RowTitlesColumn Width="160">',
            '    <Subcolumn>',
        ]
        for c_name in top_countries:
            t5_lines.append(f'      <d>{_escape(c_name)}</d>')
        t5_lines.extend([
            '    </Subcolumn>',
            '  </RowTitlesColumn>',
        ])

        for r in cohort_results:
            job_name = _escape(r.job_spec.job_name)
            cohort_c_counts: dict[str, int] = collections.defaultdict(int)
            if not r.df.empty and "countries" in r.df.columns:
                for c_list in r.df["countries"]:
                    if isinstance(c_list, list):
                        for c in set(c_list):
                            if c:
                                cohort_c_counts[str(c).strip()] += 1

            t5_lines.extend([
                f'  <YColumn Width="130" Decimals="0" Subcolumns="1">',
                f'    <Title>{job_name}</Title>',
                '    <Subcolumn>',
            ])
            for c_name in top_countries:
                t5_lines.append(f'      <d>{cohort_c_counts.get(c_name, 0)}</d>')
            t5_lines.extend([
                '    </Subcolumn>',
                '  </YColumn>',
            ])
        t5_lines.append('</Table>')
        tables_xml.append("\n".join(t5_lines))

    # -----------------------------------------------------------------------
    # Table 6: Top Publishing Institutions (OneWay / Column Graph)
    # -----------------------------------------------------------------------
    inst_totals: dict[str, int] = collections.defaultdict(int)
    for r in cohort_results:
        if not r.df.empty and "institutions" in r.df.columns:
            for i_list in r.df["institutions"]:
                if isinstance(i_list, list):
                    for inst in set(i_list):
                        if inst and str(inst).strip() and str(inst).strip() != "Unknown Institution":
                            inst_totals[str(inst).strip()] += 1

    top_institutions = [i for i, _ in collections.Counter(inst_totals).most_common(15)]
    if top_institutions:
        t_id = f"Table{table_counter}"
        table_ids.append(t_id)
        table_counter += 1

        t6_lines = [
            f'<Table ID="{t_id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
            '  <Title>Top Publishing Institutions</Title>',
            '  <RowTitlesColumn Width="240">',
            '    <Subcolumn>',
        ]
        for inst_name in top_institutions:
            t6_lines.append(f'      <d>{_escape(inst_name)}</d>')
        t6_lines.extend([
            '    </Subcolumn>',
            '  </RowTitlesColumn>',
        ])

        for r in cohort_results:
            job_name = _escape(r.job_spec.job_name)
            cohort_i_counts: dict[str, int] = collections.defaultdict(int)
            if not r.df.empty and "institutions" in r.df.columns:
                for i_list in r.df["institutions"]:
                    if isinstance(i_list, list):
                        for inst in set(i_list):
                            if inst:
                                cohort_i_counts[str(inst).strip()] += 1

            t6_lines.extend([
                f'  <YColumn Width="130" Decimals="0" Subcolumns="1">',
                f'    <Title>{job_name}</Title>',
                '    <Subcolumn>',
            ])
            for inst_name in top_institutions:
                t6_lines.append(f'      <d>{cohort_i_counts.get(inst_name, 0)}</d>')
            t6_lines.extend([
                '    </Subcolumn>',
                '  </YColumn>',
            ])
        t6_lines.append('</Table>')
        tables_xml.append("\n".join(t6_lines))

    # -----------------------------------------------------------------------
    # Table 7: Target Keywords Occurrence Dynamics (OneWay / Column Graph)
    # -----------------------------------------------------------------------
    all_kws_set: list[str] = []
    for r in cohort_results:
        for kw in r.job_spec.keywords:
            if kw not in all_kws_set:
                all_kws_set.append(kw)

    if all_kws_set:
        t_id = f"Table{table_counter}"
        table_ids.append(t_id)
        table_counter += 1

        t7_lines = [
            f'<Table ID="{t_id}" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
            '  <Title>Target Keywords Occurrence Dynamics</Title>',
            '  <RowTitlesColumn Width="180">',
            '    <Subcolumn>',
        ]
        for kw in all_kws_set:
            t7_lines.append(f'      <d>{_escape(kw)}</d>')
        t7_lines.extend([
            '    </Subcolumn>',
            '  </RowTitlesColumn>',
        ])

        for r in cohort_results:
            job_name = _escape(r.job_spec.job_name)
            t7_lines.extend([
                f'  <YColumn Width="130" Decimals="0" Subcolumns="1">',
                f'    <Title>{job_name}</Title>',
                '    <Subcolumn>',
            ])
            for kw in all_kws_set:
                col = f"kw_count_{kw}"
                cnt = int(r.df[col].sum()) if not r.df.empty and col in r.df.columns else 0
                t7_lines.append(f'      <d>{cnt}</d>')
            t7_lines.extend([
                '    </Subcolumn>',
                '  </YColumn>',
            ])
        t7_lines.append('</Table>')
        tables_xml.append("\n".join(t7_lines))

    # -----------------------------------------------------------------------
    # Per-Cohort Individual Sheets
    # -----------------------------------------------------------------------
    for idx, r in enumerate(cohort_results):
        t_id = f"Table{table_counter}"
        table_ids.append(t_id)
        table_counter += 1

        c_job_name = _escape(r.job_spec.job_name)
        c_lines = [
            f'<Table ID="{t_id}" XFormat="numbers" YFormat="replicates" Replicates="1" TableType="XY" EVFormat="AsteriskAfterNumber">',
            f'  <Title>Cohort {r.job_spec.job_id}: {c_job_name}</Title>',
            '  <XColumn Width="100" Subcolumns="1" Decimals="0">',
            '    <Title>Publication Year</Title>',
            '    <Subcolumn>',
        ]
        yr_counts = {}
        if not r.df.empty and "year" in r.df.columns:
            for y in r.df["year"].dropna():
                try:
                    iy = int(y)
                    yr_counts[iy] = yr_counts.get(iy, 0) + 1
                except (ValueError, TypeError):
                    pass

        c_years = sorted(list(yr_counts.keys())) if yr_counts else sorted_years
        for y in c_years:
            c_lines.append(f'      <d>{y}</d>')
        c_lines.extend([
            '    </Subcolumn>',
            '  </XColumn>',
            '  <YColumn Width="130" Decimals="0" Subcolumns="1">',
            f'    <Title>Publications</Title>',
            '    <Subcolumn>',
        ])
        for y in c_years:
            c_lines.append(f'      <d>{yr_counts.get(y, 0)}</d>')
        c_lines.extend([
            '    </Subcolumn>',
            '  </YColumn>',
            '</Table>',
        ])
        tables_xml.append("\n".join(c_lines))

    # -----------------------------------------------------------------------
    # Assemble Full XML Document
    # -----------------------------------------------------------------------
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    root_open = f'<GraphPadPrismFile xmlns="{PRISM_XML_NAMESPACE}" PrismXMLVersion="5.00">\n'
    created_block = (
        '<Created>\n'
        f'<OriginalVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="10.0.0" Login="ScopusTools" DateTime="{now_iso}"/>\n'
        '</Created>\n'
    )
    info_block = (
        '<InfoSequence>\n'
        '<Ref ID="Info0" Selected="1"/>\n'
        '</InfoSequence>\n'
        '<Info ID="Info0">\n'
        f'<Title>{_escape(project_title)}</Title>\n'
        '<Notes>Automated multi-cohort bibliometric extraction generated by Scopus Research Suite.</Notes>\n'
        f'<Constant><Name>Experiment Date</Name><Value>{now_date}</Value></Constant>\n'
        '<Constant><Name>Project</Name><Value>Scopus Tools Bibliometric Suite</Value></Constant>\n'
        f'<Constant><Name>Total Cohorts</Name><Value>{len(cohort_results)}</Value></Constant>\n'
        '</Info>\n'
    )

    table_seq_lines = ['<TableSequence Selected="1">']
    for tid in table_ids:
        table_seq_lines.append(f'  <Ref ID="{tid}"/>')
    table_seq_lines.append('</TableSequence>\n')
    table_seq_block = "\n".join(table_seq_lines)

    body_xml = "\n\n".join(tables_xml)
    root_close = '\n</GraphPadPrismFile>\n'

    full_xml = xml_header + root_open + created_block + info_block + table_seq_block + body_xml + root_close
    return full_xml


def export_prism_file(
    results: list[JobResult],
    output_path: str,
    project_title: str = "Scopus Bibliometric Mass Export",
) -> None:
    """Write a GraphPad Prism XML project (.pzfx) to disk.

    Parameters
    ----------
    results:
        List of completed JobResult objects.
    output_path:
        Destination file path for the .pzfx file.
    project_title:
        Title recorded in the Prism project Info sheet.
    """
    xml_content = generate_prism_xml(results, project_title=project_title)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(xml_content)
    logger.info("Successfully exported GraphPad Prism project to %s", output_path)
