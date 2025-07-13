import itertools
import logging
from difflib import SequenceMatcher
from typing import Optional

import gravis as gv
import matplotlib as mpl
import networkx as nx
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from annotated_text import annotated_text

STATISTICS_URL_DATASET = (
    'https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002522_00005024.csv'
)
STATISTICS_URL = (
    'https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002522_00005043.csv'
)
STATISTICS_METADATA_URL = 'https://opendata.swiss/de/dataset/web-analytics-des-datenkatalogs-des-kantons-zurich/resource/6af2b395-47ce-491c-9083-8cf58e67aca9'
METADATA_URL = 'https://www.web.statistik.zh.ch/ogd/daten/zhweb.json'
GITHUB_URL = 'https://github.com/fbardos/ktzh_ogd_statistics'
DEFAULT_EXCLUDE_KEYWORDS = {
    'ogd',
    'kanton_zuerich',
    'bezirke',
    'gemeinden',
    'statistik.info',
}
OGD_METADATA_URL = 'https://www.web.statistik.zh.ch/ogd/datenkatalog/standalone/'
VALUE_COLUMNS = {
    'anzahl_klicks_dataset': 'Klicks Datensatz',
    'anzahl_besuchende_dataset': 'Besuchende Datensatz',
    'anzahl_klicks': 'Klicks Ressourcen',
    'anzahl_besuchende': 'Besuchende Ressourcen',
    'anzahl_downloads': 'Downloads Files',
}


@st.cache_data(ttl=60 * 60 * 2)  # Cache for 2 hours
def compare_keywords(
    data: pd.DataFrame,
    exclude_keywords: set,
    similarity: float,
    threshold_long: int,
    weight_factor: float,
):
    # Alternative approach for comparing similarity (much faster than iterrows)
    G = nx.Graph()
    df_cartesian = data.merge(data, how='cross', suffixes=('_x', '_y'))
    df_cartesian = df_cartesian[df_cartesian['id_x'] < df_cartesian['id_y']]
    df_cartesian['keyword_x'] = df_cartesian['keyword_x'].apply(set)
    df_cartesian['keyword_y'] = df_cartesian['keyword_y'].apply(set)
    df_cartesian['_sub_keyword_x'] = df_cartesian['keyword_x'] - exclude_keywords
    df_cartesian['_sub_keyword_y'] = df_cartesian['keyword_y'] - exclude_keywords
    df_cartesian['_sub_keyword_x'] = df_cartesian['_sub_keyword_x'].apply(list)
    df_cartesian['_sub_keyword_y'] = df_cartesian['_sub_keyword_y'].apply(list)
    df_cartesian['weight'] = df_cartesian.apply(
        lambda x: SequenceMatcher(
            None, x['_sub_keyword_x'], x['_sub_keyword_y']
        ).ratio(),
        axis=1,
    )
    df_cartesian = df_cartesian[
        (df_cartesian['weight'] > similarity)
        & (df_cartesian['avg_long_x'] >= threshold_long)
        & (df_cartesian['avg_long_y'] >= threshold_long)
    ]
    df_cartesian.apply(
        lambda x: G.add_edge(x['id_x'], x['id_y'], weight=x['weight'] * weight_factor),
        axis=1,
    )
    return G


def main(
    data_stat: pd.DataFrame,
    data_meta: pd.DataFrame,
    value_col: Optional[str],
    avg_short_days: int = 30,
    avg_long_days: int = 360,
    exclude_keywords: set = DEFAULT_EXCLUDE_KEYWORDS,
    exclude_orgs: list = [],
    include_orgs: list = [],
    include_keywords: list = [],
    bigger_than_similarity: float = 0.0,
    threshold_avg_long: int = 1,
    weight_factor: float = 2.0,
    spring_k: Optional[float] = None,
    scale: Optional[float] = None,
):

    if value_col is None:
        value_col = 'anzahl_klicks'

    STATISTICS_VALUE_COL = value_col
    AVG_SHORT_DAYS = avg_short_days
    AVG_LONG_DAYS = avg_long_days
    BIGGER_THAN_SIMILARITY = bigger_than_similarity
    THRESHOLD_AVG_LONG = threshold_avg_long
    EXCLUDE_KEYWORDS = exclude_keywords

    col_label = VALUE_COLUMNS[STATISTICS_VALUE_COL]

    prog = st.progress(0, text='Ausführen der grundlegenden Datentransformation...')
    if len(exclude_orgs) > 0:
        data_meta = data_meta[~data_meta['publisher'].isin(exclude_orgs)]
    if len(include_orgs) > 0:
        data_meta = data_meta[data_meta['publisher'].isin(include_orgs)]
    if len(include_keywords) > 0:
        data_meta = data_meta[
            data_meta['keyword']
            .fillna('')
            .apply(list)
            .apply(lambda val: any([k in val for k in include_keywords]))
        ]
    logging.info('Calculate statics...')
    prog.progress(0.1, text='Do basic table operations...')
    df = data_meta.merge(data_stat, left_on='id', right_on='datensatz_id', how='outer')
    df['diff_days'] = (pd.to_datetime('now') - pd.to_datetime(df['datum'])).dt.days

    # Iterate over variables and calculate average values
    for col_name in VALUE_COLUMNS.keys():
        df_avg_short = (
            df[df['diff_days'] <= AVG_SHORT_DAYS][['id', col_name]]
            .groupby('id')
            .sum()
            .reset_index()
            .rename(columns={col_name: f'avg_short__{col_name}'})
        )
        df_avg_long = (
            df[df['diff_days'] <= AVG_LONG_DAYS][['id', col_name]]
            .groupby('id')
            .sum()
            .reset_index()
            .rename(columns={col_name: f'avg_long__{col_name}'})
        )
        df_avg_long[f'avg_long__{col_name}'] = df_avg_long[f'avg_long__{col_name}'] / (
            AVG_LONG_DAYS / AVG_SHORT_DAYS
        )
        data_meta = data_meta.merge(df_avg_short, on='id', how='left').merge(
            df_avg_long, on='id', how='left'
        )

    # Select data column in dataframe
    data_meta['avg_short'] = data_meta[f'avg_short__{STATISTICS_VALUE_COL}']
    data_meta['avg_long'] = data_meta[f'avg_long__{STATISTICS_VALUE_COL}']

    _normalized_colors = mpl.colors.Normalize(vmin=0, vmax=2, clip=True)
    _colors = sns.color_palette('coolwarm', as_cmap=True)
    _colors.set_under('#3b4cc0')
    _colors.set_over('#3c4ec2')
    data_meta = (
        data_meta.assign(avg_ratio=lambda x: x['avg_short'] / x['avg_long'])
        .assign(avg_ratio=lambda x: x['avg_ratio'].fillna(1))
        .assign(avg_ratio_normalized=lambda x: _normalized_colors(x['avg_ratio']))
        .assign(avg_short=lambda x: x['avg_short'].fillna(0))
        .assign(avg_long=lambda x: x['avg_long'].fillna(0))
        .assign(keyword=lambda x: x['keyword'].fillna('').apply(list))
    )
    data_meta['avg_ratio_color'] = data_meta.apply(
        lambda x: mpl.colors.rgb2hex(_colors(x['avg_ratio_normalized'])), axis=1
    )

    # Building graph
    logging.info('Build graph...')
    prog.progress(
        0.4, text='Keywords aller Datensätze werden miteinander verglichen...'
    )
    G = compare_keywords(
        data=data_meta,
        exclude_keywords=EXCLUDE_KEYWORDS,
        similarity=BIGGER_THAN_SIMILARITY,
        threshold_long=THRESHOLD_AVG_LONG,
        weight_factor=weight_factor,
    )

    # Add info to nodes
    # Normalize node size (min value = 2, multiply by 30)
    _df = data_meta[['id', 'avg_short']].copy().set_index('id')
    _df_node = (_df - _df.min()) * 80 / (_df.max() - _df.min()) + 8
    _node_size = _df_node.to_dict()['avg_short']
    nx.set_node_attributes(G, _node_size, 'size')
    _df_label = (_df - _df.min()) * 40 / (_df.max() - _df.min()) + 12
    _label_size = _df_label.to_dict()['avg_short']
    nx.set_node_attributes(G, _label_size, 'label_size')

    # Set border color and size according to amount of klicks
    _df = data_meta[['id', 'avg_ratio_color']].copy().set_index('id')
    _border_color = _df.to_dict()['avg_ratio_color']
    nx.set_node_attributes(G, _border_color, 'color')

    # Set colors based on publisher
    _publisher = data_meta['publisher'].value_counts().to_dict()
    _colors = sns.color_palette('deep', len(_publisher)).as_hex()
    for k in _publisher.keys():
        _publisher[k] = _colors.pop()
    _df = data_meta[['id', 'publisher']].copy().set_index('id')
    _df = _df['publisher'].map(_publisher)
    data_meta['publisher_color'] = data_meta['publisher'].map(
        _publisher
    )  # write back to original df for statistics table
    _node_color = _df.to_dict()
    nx.set_node_attributes(G, _node_color, 'label_color')

    # Generate metadata for gravis
    prog.progress(
        0.7,
        text=f'Der Graph hat {G.number_of_nodes()} Nodes und {G.number_of_edges()} Udges. Generiere Metadaten...',
    )
    _hover = {}
    for _, row in data_meta.iterrows():
        _hover[row['id']] = (
            f"<b>{row['title']}</b><br>"
            f"<i>{row['description']}</i><br><br>"
            "<table>"
            "<tr>"
            "<td><b>ID</b></td>"
            f"<td>{row['id']}</td>"
            "</tr>"
            "<tr>"
            "<td><b>Organisation</b></td>"
            f"<td>{row['publisher']}</td>"
            "</tr>"
            "<tr>"
            f"<td><b>{col_label} ({AVG_SHORT_DAYS} Tage)</b></td>"
            f"<td>{int(row['avg_short'])} {'▲' if row['avg_short'] > row['avg_long'] else '▼'}</td>"
            "</tr>"
            "<tr>"
            f"<td><b>{col_label} ({AVG_LONG_DAYS} Tage, pro {AVG_SHORT_DAYS} Tage)</b></td>"
            f"<td>{round(row['avg_long'], 1)}</td>"
            "</tr>"
            "</table>"
            "<br>"
            "<b>Link Datensatz</b><br>"
            f"<a href='{OGD_METADATA_URL}datasets/{row['identifier']}' target='_blank'>{row['identifier']}</a><br>"
            "<br>"
            "<b>Keywords</b><br>"
            f"{', '.join(row['keyword'])}"
        )
    nx.set_node_attributes(G, _hover, 'hover')

    # Relabel nodes, needs to be executed last
    _node_title = data_meta[['id', 'title']].set_index('id').to_dict()['title']
    nx.relabel_nodes(G, _node_title, copy=False)

    logging.info(
        f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges'
    )
    logging.info('Calculate Graph layout...')
    prog.progress(0.8, text='Simluation des Graphen in 200 iterationen...')

    # Set defaults for empty k or scale
    if spring_k is None:
        spring_k = 1 / G.number_of_nodes() ** (1 / 3)
    if scale is None:

        # Use a function from regression (quadratic) instead of previous approach:
        # https://www.wolframalpha.com/input?i=quadratic+fit+%7B0%2C500%7D%2C%7B10%2C1000%7D%2C%7B254%2C14000%7D%2C%7B500%2C20000%7D
        scale = max(
            0,
            min(
                20_000,
                -0.0580631 * G.number_of_nodes() ** 2
                + 68.2061 * G.number_of_nodes()
                + 414.532,
            ),
        )

    pos = nx.spring_layout(
        G, k=spring_k, iterations=200, center=(0, 0), scale=scale, seed=42
    )
    for name, (x, y) in pos.items():
        node = G.nodes[name]
        node['x'] = x
        node['y'] = y

    # gJGF hacking, additional informaiton
    prog.progress(0.9, text='HTML für Graph generieren...')
    gjgf = gv.convert.networkx_to_gjgf(G)
    gjgf['graph']['metadata'] = {}
    gjgf['graph']['metadata']['background_color'] = '#0D0D0D'
    gjgf['graph']['metadata']['node_color'] = '#BE3144'
    gjgf['graph']['metadata']['node_opacity'] = 40
    gjgf['graph']['metadata']['node_label_color'] = '#BE3144'
    gjgf['graph']['metadata']['edge_color'] = '#1f787d'
    gjgf['graph']['metadata']['edge_opacity'] = 60

    fig = gv.d3(
        gjgf,
        graph_height=1200,
        show_menu=False,
        show_menu_toggle_button=False,
        show_details=False,
        show_details_toggle_button=False,
        layout_algorithm_active=False,
        edge_curvature=0.3,
        edge_size_data_source='weight',
        node_hover_tooltip=True,
        zoom_factor=0.75,
    )
    prog.progress(0.95, text='Berechnen Organisationsstatistiken...')
    df_stat_org = (
        data_meta[
            [
                'publisher',
                'publisher_color',
                'avg_short__anzahl_klicks_dataset',
                'avg_short__anzahl_besuchende_dataset',
                'avg_short__anzahl_klicks',
                'avg_short__anzahl_besuchende',
                'avg_short__anzahl_downloads',
            ]
        ]
        .groupby(['publisher', 'publisher_color'])
        .agg(
            size=('avg_short__anzahl_klicks', 'size'),
            sum_dk=('avg_short__anzahl_klicks_dataset', 'sum'),
            sum_dv=('avg_short__anzahl_besuchende_dataset', 'sum'),
            sum=('avg_short__anzahl_klicks', 'sum'),
            sum_v=('avg_short__anzahl_besuchende', 'sum'),
            sum_d=('avg_short__anzahl_downloads', 'sum'),
        )
        .reset_index()
        .sort_values(by='sum', ascending=False)
        .assign(sum_dk=lambda x: x['sum_dk'].astype(int))
        .assign(sum_dv=lambda x: x['sum_dv'].astype(int))
        .assign(sum=lambda x: x['sum'].astype(int))
        .assign(sum_v=lambda x: x['sum_v'].astype(int))
        .assign(sum_d=lambda x: x['sum_d'].astype(int))
        .rename(
            columns={
                'publisher': 'Organisation',
                'publisher_color': 'Farbe',
                'size': 'Anzahl Datensätze',
                'sum_dk': f'Klicks Datensatz',
                'sum_dv': f'Besuchende Datensatz',
                'sum': f'Klicks Ressourcen',
                'sum_v': f'Besuchende Ressourcen',
                'sum_d': f'Downloads Ressourcen',
            }
        )
    )
    df_stat_dataset = (
        data_meta[
            [
                'title',
                'identifier',
                'publisher',
                'avg_short__anzahl_klicks_dataset',
                'avg_short__anzahl_besuchende_dataset',
                'avg_short__anzahl_klicks',
                'avg_short__anzahl_besuchende',
                'avg_short__anzahl_downloads',
            ]
        ]
        .assign(url=lambda x: OGD_METADATA_URL + 'datasets/' + x["identifier"], axis=1)
        .drop(columns=['identifier'])
        .groupby(['title', 'url', 'publisher'])
        .agg(
            sum_dk=('avg_short__anzahl_klicks_dataset', 'sum'),
            sum_dv=('avg_short__anzahl_besuchende_dataset', 'sum'),
            sum=('avg_short__anzahl_klicks', 'sum'),
            sum_v=('avg_short__anzahl_besuchende', 'sum'),
            sum_d=('avg_short__anzahl_downloads', 'sum'),
        )
        .reset_index()
        .sort_values(by='sum', ascending=False)
        .assign(sum=lambda x: x['sum'].astype(int))
        .assign(sum_v=lambda x: x['sum_v'].astype(int))
        .rename(
            columns={
                'title': 'Datensatz',
                'publisher': 'Organisation',
                'sum_dk': f'Klicks Datensatz',
                'sum_dv': f'Besuchende Datensatz',
                'sum': f'Klicks Ressourcen',
                'sum_v': f'Besuchende Ressourcen',
                'sum_d': f'Downloads Ressourcen',
            }
        )
    )

    prog.empty()
    return fig, df_stat_org, df_stat_dataset, G.number_of_nodes(), G.number_of_edges()


def intro_text(days_short: int = 30):
    return f"""
        Diese Applikation visualisiert die Zugriffsstatistik der OGD-Datensätze
        des [Metadatenkatalogs des Kantons Zürich]({OGD_METADATA_URL}).
        Dabei wird ein Graph generiert, der jeweils die Ähnlichkeit zweier Datensätze darstellt.
        Die Zugriffsstatistik bildet Zugriffe auf den Datenkatalog des Kantons Zürich ab, sowohl für Ressourcen als auch für Datensätze.
        Als Zugriffe können Klicks (Seitenaufrufe), Besucher oder Downloads ausgewählt werden.
        Nicht enthalten Zugriffe auf anderen OGD-Katalogen wie [opendata.swiss](http://opendata.swiss).

        * Verwenden zwei Datensätze ähnliche Keywords, dann stehen sie näher beeinander (Spring Layout).
        * Die **Grösse der Nodes** repräsentiert die Anzahl der Zugriffe der letzten `{days_short}` Tage.
        * Die **Farbe des Nodes** gibt an, ob die Zugriffe in kürzerer Zeit zugenommen (:red[rot]) oder abgenommen (:blue[blau]) haben.
        * Die **Schriftfarbe der Nodes** gibt die Organisation an, die den Datensatz publiziert hat.
        * Die **Dicke der Edges** repräsentiert die Ähnlichkeit der Keywords zweier Datensätze.
    """


if __name__ == '__main__':
    # Get basic data (for dataset and resource, then merge them)
    logging.info('Get data from KTZH metacatalogue...')
    data_stat_dataset = pd.read_csv(STATISTICS_URL_DATASET).rename(
        columns={
            'anzahl_klicks': 'anzahl_klicks_dataset',
            'anzahl_besuchende': 'anzahl_besuchende_dataset',
        }
    )
    data_stat = pd.read_csv(STATISTICS_URL)

    # Data regroup on level dataset (not resource)
    data_stat = (
        data_stat[
            [
                'datum',
                'datensatz_id',
                'publisher',
                'datensatz_titel',
                'anzahl_klicks',
                'anzahl_besuchende',
                'anzahl_downloads',
            ]
        ]
        .groupby(['datum', 'datensatz_id', 'publisher', 'datensatz_titel'])
        .sum()
        .reset_index()
    )
    data_stat['anzahl_downloads'] = data_stat['anzahl_downloads'].astype(int)

    # merge both dataframes and recalculate values, afterwards regroup
    data_stat = pd.concat([data_stat, data_stat_dataset], ignore_index=True)
    data_stat = (
        data_stat.groupby(['datum', 'datensatz_id', 'publisher', 'datensatz_titel'])
        .sum()
        .reset_index()
    )

    response_meta = requests.get(METADATA_URL).json()
    data_meta = pd.DataFrame.from_dict(response_meta['dataset'])
    data_meta = data_meta[
        ['identifier', 'title', 'description', 'keyword', 'publisher']
    ]
    available_keywords = set(
        itertools.chain.from_iterable(
            data_meta[data_meta['keyword'].notnull()]['keyword']
        )
    )
    data_meta['id'] = data_meta['identifier'].str.extract(r'(\d+)@.*').astype(int)
    data_meta['publisher'] = data_meta.apply(lambda x: x['publisher'][0], axis=1)
    available_orgs = sorted(data_meta['publisher'].unique())

    logging.basicConfig(level=logging.INFO)
    st.set_page_config(layout="wide")
    st.title('OGD Kanton Zürich Zugriffsstatistik')
    header_col1, header_col2 = st.columns((0.7, 0.3), gap='medium')
    intro = header_col1.markdown(intro_text(30))
    header_col2.markdown(
        f"""
        Quellen:
        * [OGD Metadatenkatalog Kanton Zürich]({OGD_METADATA_URL})
        * [Datensätze Metadatenkatalog (API)]({METADATA_URL})
        * [Zugriffs-Statistik (Ebene Datensatz)]({STATISTICS_URL_DATASET})
        * [Zugriffs-Statistik (Ebene Resource)]({STATISTICS_URL})
        * [Link Datensatz Metadatenkatalog]({STATISTICS_METADATA_URL})
        * [Github-Repo]({GITHUB_URL})
    """
    )

    container = st.container(border=True)
    input_col1, input_col2 = container.columns(2, gap='medium')
    input_col1.subheader('Filter')
    input_value_col = input_col1.radio(
        label='Anzuzeigende Werte',
        options=list(VALUE_COLUMNS.keys()),
        format_func=lambda x: VALUE_COLUMNS[x],
    )
    timespan = input_col1.slider(
        'Vergleich Zugriffszahlen (in Tagen)',
        1,
        360,
        (30, 180),
    )
    input_exclude_orgs = input_col1.multiselect(
        'Organisationen exkludieren:',
        available_orgs,
    )
    input_include_orgs = input_col1.multiselect(
        'Organisationen auswählen:',
        available_orgs,
    )
    input_include_keywords = input_col1.multiselect(
        'Keywords auswählen:',
        sorted(available_keywords),
    )

    input_col2.subheader('Generierung Graph')
    intro.write(intro_text(timespan[0]))
    input_thresold_avg_long = input_col2.slider(
        f'Durchschnittliche Zugriffe (mindestens, letzte {timespan[1]} Tage)',
        1,
        100,
        1,
    )
    excl_keywords = input_col2.multiselect(
        'Auszuschliessende Keywords (beim Vergleich der Ähnlichkeit von Keywords zweier Datensätze)',
        sorted(available_keywords),
        list(DEFAULT_EXCLUDE_KEYWORDS),
    )
    with input_col2.expander('Advanced') as exp:

        input_bigger_than_similarity = st.slider(
            'Erforderliche Ähnlichkeit für Darstellung des Edges (grösser als, Werte von 0 bis 1)',
            0.0,
            1.0,
            0.0,
            step=0.01,
        )
        input_scale = st.number_input(
            'Skalierung des Graphen (je grösser, desto grösser wird die Karte). Werte zw. 200 und 30000. Default `-0.0580631 x^2 + 68.2061 x + 414.532`',
            min_value=200,
            max_value=30_000,
            value=None,
        )
        input_spring_k = st.number_input(
            'Optimale Distanz zwischen Nodes (Werte zwischen 0.01 und 1). Je grösser, desto weiter liegen Nodes auseinander. Default: `1/anzahl_nodes**(1/3)`',
            min_value=0.01,
            max_value=1.0,
            value=None,
        )

    fig, df_stat_out_org, df_stat_out_dataset, cnt_nodes, cnt_edges = main(
        data_stat=data_stat,
        data_meta=data_meta,
        value_col=input_value_col,
        avg_long_days=timespan[1],
        avg_short_days=timespan[0],
        exclude_keywords=set(excl_keywords),
        bigger_than_similarity=input_bigger_than_similarity,
        threshold_avg_long=input_thresold_avg_long,
        exclude_orgs=input_exclude_orgs,
        include_orgs=input_include_orgs,
        include_keywords=input_include_keywords,
        spring_k=input_spring_k,
        scale=input_scale,
    )

    st.subheader('Graph')
    annotated_text(
        (f'{cnt_nodes}', 'Nodes'),
        ' ',
        (f'{cnt_edges}', 'Edges'),
    )

    _html = fig.to_html_standalone()
    st.components.v1.html(_html, height=1208)
    st.subheader('Zugriffsstatistik Organisation')
    st.markdown(
        f'Die nachfolgende Tabelle zeigt die Zugriffsstatistik der publizierenden Organisationen der letzten'
        f'`{timespan[0]}` Tage unter Berücksichtigung der gesetzten Filter.'
    )
    st.dataframe(
        df_stat_out_org.style.applymap(
            lambda col: f"background-color: {col}", subset=['Farbe']
        ),
        hide_index=True,
    )
    st.subheader('Zugriffsstatistik Datensatz')
    st.markdown(
        f'Die nachfolgende Tabelle zeigt die Zugriffsstatistik der einzelnen Datensätze der letzten'
        f'`{timespan[0]}` Tage unter Berücksichtigung der gesetzten Filter.'
    )
    st.dataframe(
        df_stat_out_dataset,
        column_config=dict(
            url=st.column_config.LinkColumn('URL', display_text='Link'),
        ),
        hide_index=True,
    )
