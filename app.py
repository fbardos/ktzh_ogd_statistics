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

STATISTICS_URL = 'https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002522_00005024.csv'
STATISTICS_METADATA_URL = 'https://opendata.swiss/de/dataset/web-analytics-des-datenkatalogs-des-kantons-zurich/resource/c72eda06-befb-4b21-bc39-75340f7546cb'
METADATA_URL = 'https://www.web.statistik.zh.ch/ogd/daten/zhweb.json'
GITHUB_URL = 'https://github.com/fbardos/ktzh_ogd_statistics'
DEFAULT_EXCLUDE_KEYWORDS = {'ogd', 'kanton_zuerich', 'bezirke', 'gemeinden', 'statistik.info'}
OGD_METADATA_URL = 'https://www.web.statistik.zh.ch/ogd/datenkatalog/standalone/'


# Get basic data
logging.info('Get data from KTZH metacatalogue...')
data_stat = pd.read_csv(STATISTICS_URL)
response_meta = requests.get(METADATA_URL).json()
data_meta = pd.DataFrame.from_dict(response_meta['dataset'])
data_meta = data_meta[['identifier', 'title', 'description', 'keyword', 'publisher']]
available_keywords = set(itertools.chain.from_iterable(data_meta[data_meta['keyword'].notnull()]['keyword']))
data_meta['id'] = data_meta['identifier'].str.extract(r'(\d+)@.*').astype(int)
data_meta['publisher'] = data_meta.apply(lambda x: x['publisher'][0], axis=1)
available_orgs = sorted(data_meta['publisher'].unique())


@st.cache_data(ttl=60*60*2)  # Cache for 2 hours
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
    df_cartesian['weight'] = df_cartesian.apply(lambda x: SequenceMatcher(None, x['_sub_keyword_x'], x['_sub_keyword_y']).ratio(), axis=1)
    df_cartesian = df_cartesian[(df_cartesian['weight'] > similarity) & (df_cartesian['avg_long_x'] >= threshold_long) & (df_cartesian['avg_long_y'] >= threshold_long)]
    df_cartesian.apply(lambda x: G.add_edge(x['id_x'], x['id_y'], weight=x['weight']*weight_factor), axis=1)
    return G


def main(
    data_stat: pd.DataFrame,
    data_meta: pd.DataFrame,
    avg_short_days: int = 30,
    avg_long_days: int = 180,
    exclude_keywords: set = DEFAULT_EXCLUDE_KEYWORDS,
    exclude_orgs: list = [],
    include_orgs: list = [],
    bigger_than_similarity: float = 0.0,
    threshold_avg_long: int = 1,
    weight_factor: float = 2.0,
    spring_k: Optional[float] = None,
    scale: int = 10_000,
):
    
    STATISTICS_COL = 'anzahl_klicks'
    AVG_SHORT_DAYS = avg_short_days 
    AVG_LONG_DAYS = avg_long_days
    BIGGER_THAN_SIMILARITY = bigger_than_similarity
    THRESHOLD_AVG_LONG = threshold_avg_long
    EXCLUDE_KEYWORDS = exclude_keywords
    
    prog = st.progress(0, text='Ausführen der grundlegenden Datentransformation...')
    if len(exclude_orgs) > 0:
        data_meta = data_meta[~data_meta['publisher'].isin(exclude_orgs)]
    if len(include_orgs) > 0:
        data_meta = data_meta[data_meta['publisher'].isin(include_orgs)]
    logging.info('Calculate statics...')
    prog.progress(0.1, text='Do basic table operations...')
    df = data_meta.merge(data_stat, left_on='id', right_on='datensatz_id', how='outer')
    df['diff_days'] = (pd.to_datetime('now') - pd.to_datetime(df['datum'])).dt.days
    df_avg_short = (
        df[df['diff_days'] <= AVG_SHORT_DAYS][['id', STATISTICS_COL]]
        .groupby('id')
        .sum()
        .reset_index()
        .rename(columns={STATISTICS_COL: 'avg_short'})
    )
    df_avg_long = (
        df[df['diff_days'] <= AVG_LONG_DAYS][['id', STATISTICS_COL]]
        .groupby('id')
        .sum()
        .reset_index()
        .rename(columns={STATISTICS_COL: 'avg_long'})
        .assign(avg_long=lambda x: x['avg_long'] / (AVG_LONG_DAYS / AVG_SHORT_DAYS))
    )
    data_meta = (
        data_meta
        .merge(df_avg_short, on='id', how='left')
        .merge(df_avg_long, on='id', how='left')
    )
    _normalized_colors = mpl.colors.Normalize(vmin=0, vmax=2, clip=True)
    _colors = sns.color_palette('coolwarm', as_cmap=True)
    _colors.set_under('#3b4cc0')
    _colors.set_over('#3c4ec2')
    data_meta = (
        data_meta
        .assign(avg_ratio=lambda x: x['avg_short'] / x['avg_long'])
        .assign(avg_ratio=lambda x: x['avg_ratio'].fillna(1))
        .assign(avg_ratio_normalized=lambda x: _normalized_colors(x['avg_ratio']))
        .assign(avg_short=lambda x: x['avg_short'].fillna(0))
        .assign(avg_long=lambda x: x['avg_long'].fillna(0))
        .assign(keyword=lambda x: x['keyword'].fillna('').apply(list))
    )
    data_meta['avg_ratio_color'] = data_meta.apply(lambda x: mpl.colors.rgb2hex(_colors(x['avg_ratio_normalized'])), axis=1)

    # Building graph
    logging.info('Build graph...')
    prog.progress(0.4, text='Keywords aller Datensätze werden miteinander verglichen...')
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
    data_meta['publisher_color'] = data_meta['publisher'].map(_publisher)  # write back to original df for statistics table
    _node_color = _df.to_dict()
    nx.set_node_attributes(G, _node_color, 'label_color')

    # Generate metadata for gravis
    prog.progress(0.7, text=f'Der Graph hat {G.number_of_nodes()} Nodes und {G.number_of_edges()} Udges. Generiere Metadaten...')
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
            f"<td><b>Klicks ({AVG_SHORT_DAYS} Tage)</b></td>"
            f"<td>{int(row['avg_short'])} {'▲' if row['avg_short'] > row['avg_long'] else '▼'}</td>"
            "</tr>"
            "<tr>"
            f"<td><b>Klicks ({AVG_LONG_DAYS} Tage, pro {AVG_SHORT_DAYS} Tage)</b></td>"
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

    logging.info(f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    logging.info('Calculate Graph layout...')
    prog.progress(0.8, text='Simluation des Graphen in 200 iterationen...')
    pos = nx.spring_layout(G, k=spring_k, iterations=200, center=(0, 0), scale=scale, seed=42)
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
        data_meta[['publisher', 'publisher_color', 'avg_short']]
        .groupby(['publisher', 'publisher_color'])
        .agg(
            size=('avg_short', 'size'),
            sum=('avg_short', 'sum'),
        )
        .reset_index()
        .sort_values(by='sum', ascending=False)
        .assign(sum=lambda x: x['sum'].astype(int))
        .rename(columns={
            'publisher': 'Organisation',
            'publisher_color': 'Farbe',
            'size': 'Anzahl Datensätze',
            'sum': f'Klicks (letzte {AVG_SHORT_DAYS} Tage)',
        })
    )
    df_stat_dataset = (
        data_meta[['title', 'identifier', 'publisher', 'avg_short']]
        .assign(url=lambda x: OGD_METADATA_URL + 'datasets/' + x["identifier"], axis=1)
        .drop(columns=['identifier'])
        .groupby(['title', 'url', 'publisher'])
        .agg(
            sum=('avg_short', 'sum'),
        )
        .reset_index()
        .sort_values(by='sum', ascending=False)
        .assign(sum=lambda x: x['sum'].astype(int))
        .rename(columns={
            'title': 'Datensatz',
            'publisher': 'Organisation',
            'sum': f'Klicks (letzte {AVG_SHORT_DAYS} Tage)',
        })
    )
    
    prog.empty()
    return fig, df_stat_org, df_stat_dataset, G.number_of_nodes(), G.number_of_edges()


def intro_text(days_short: int = 30):
    return f"""
        Diese Applikation visualisiert die Zugriffsstatistik der OGD-Datensätze
        des [Metadatenkatalogs des Kantons Zürich]({OGD_METADATA_URL}).
        Dabei wird ein Graph generiert, der die Ähnlichkeit zweier Datensätze darstellt.
        Die Zugriffsstatistik bildet Zugriffe (Klicks) auf den Datenkatalog des Kantons Zürich ab. Nicht enthalten
        sind direkte File-Zugriffe oder Zugriffe von anderen Katalogen wie [opendata.swiss](http://opendata.swiss).

        * Verwenden zwei Datensätze ähnliche Keywords, dann stehen sie näher beeinander (Spring Layout).
        * Die **Grösse der Nodes** repräsentiert die Anzahl der Zugriffe der letzten `{days_short}` Tage.
        * Die **Farbe des Nodes** gibt an, ob die Zugriffe in kürzerer Zeit zugenommen (:red[rot]) oder abgenommen (:blue[blau]) haben.
        * Die **Schriftfarbe der Nodes** gibt die Organisation an, die den Datensatz publiziert hat.
        * Die **Dicke der Edges** repräsentiert die Ähnlichkeit der Keywords zweier Datensätze.
    """
    

logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")
st.title('OGD Kanton Zürich Zugriffsstatistik')
header_col1, header_col2 = st.columns((0.7, 0.3), gap='medium')
intro = header_col1.markdown(intro_text(30))
header_col2.markdown(f"""
    Quellen:
    * [OGD Metadatenkatalog Kanton Zürich]({OGD_METADATA_URL})
    * [Datensätze Metadatenkatalog (API)]({METADATA_URL})
    * [Zugriffs-Statistik (Ebene Datensatz)]({STATISTICS_URL})
    * [Link Datensatz Metadatenkatalog]({STATISTICS_METADATA_URL})
    * [Github-Repo]({GITHUB_URL})
""")

container = st.container(border=True)
input_col1, input_col2 = container.columns(2, gap='medium')
input_col1.subheader('Filter')
timespan = input_col1.slider(
    'Vergleich Zugriffszahlen (in Tagen)',
    1, 180, (30, 180),
)
input_exclude_orgs = input_col1.multiselect(
    'Organisationen exkludieren:',
    available_orgs,
)
input_include_orgs = input_col1.multiselect(
    'Organisationen auswählen:',
    available_orgs,
)

input_col2.subheader('Generierung Graph')
intro.write(intro_text(timespan[0]))
input_thresold_avg_long = input_col2.slider(
    f'Durchschnittliche Zugriffe (mindestens, letzte {timespan[1]} Tage)',
    1, 100, 1,
)
excl_keywords = input_col2.multiselect(
    'Auszuschliessende Keywords (beim Vergleich der Ähnlichkeit von Keywords zweier Datensätze)',
    sorted(available_keywords),
    list(DEFAULT_EXCLUDE_KEYWORDS),
)
with input_col2.expander('Advanced') as exp:

    input_bigger_than_similarity = st.slider(
        'Erforderliche Ähnlichkeit für Darstellung des Edges (grösser als, Werte von 0 bis 1)',
        0.0, 1.0, 0.0,
        step=0.01,
    )
    input_scale = st.slider(
        'Skalierung des Graphen (je grösser, desto grösser wird die Karte)',
        500, 30_000, 10_000,
        step=500,
    )
    input_spring_k = st.number_input(
        'Optimale Distanz zwischen Nodes (Werte zwischen 0.01 und 1). Je grösser, desto weiter liegen Nodes auseinander. Default: `1 / sqrt(anzahl_nodes)`',
        min_value=0.01,
        max_value=1.0,
        value=None,
        # placeholder='Default = 1 / sqrt(anzahl_nodes)',
    )

fig, df_stat_out_org, df_stat_out_dataset, cnt_nodes, cnt_edges = main(
    data_stat=data_stat,
    data_meta=data_meta,
    avg_long_days=timespan[1],
    avg_short_days=timespan[0],
    exclude_keywords=set(excl_keywords),
    bigger_than_similarity=input_bigger_than_similarity,
    threshold_avg_long=input_thresold_avg_long,
    exclude_orgs=input_exclude_orgs,
    include_orgs=input_include_orgs,
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
    df_stat_out_org.style.applymap(lambda col: f"background-color: {col}", subset=['Farbe']),
    hide_index=True
)
st.subheader('Zugriffsstatistik Datensatz')
st.dataframe(
    df_stat_out_dataset,
    column_config=dict(
        url=st.column_config.LinkColumn(
            'URL',
            display_text='Link'
        ),
    ),
    hide_index=True
)
