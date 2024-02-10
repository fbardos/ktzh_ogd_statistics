import itertools
import logging
from collections import Counter
from difflib import SequenceMatcher

import gravis as gv
import matplotlib as mpl
import networkx as nx
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

STATISTICS_URL = 'https://www.web.statistik.zh.ch/ogd/daten/ressourcen/KTZH_00002522_00005024.csv'
METADATA_URL = 'https://www.web.statistik.zh.ch/ogd/daten/zhweb.json'
GITHUB_URL = 'https://github.com/fbardos/ktzh_ogd_statistics'
DEFAULT_EXCLUDE_KEYWORDS = {'ogd', 'kanton_zuerich', 'bezirke', 'gemeinden'}

# Get basic data
logging.info('Get data from KTZH metacatalogue...')
data_stat = pd.read_csv(STATISTICS_URL)
response_meta = requests.get(METADATA_URL).json()
data_meta = pd.DataFrame.from_dict(response_meta['dataset'])
data_meta = data_meta[['identifier', 'title', 'description', 'keyword', 'publisher']]
available_keywords = set(itertools.chain.from_iterable(data_meta[data_meta['keyword'].notnull()]['keyword']))

def main(
    data_stat: pd.DataFrame,
    data_meta: pd.DataFrame,
    avg_short_days: int = 30,
    avg_long_days: int = 180,
    exclude_keywords: set = DEFAULT_EXCLUDE_KEYWORDS,
    bigger_than_similarity: float = 0.0,
    threshold_avg_long: int = 1,
    scale: int = 10_000,
):
    
    STATISTICS_COL = 'anzahl_klicks'
    AVG_SHORT_DAYS = avg_short_days 
    AVG_LONG_DAYS = avg_long_days
    BIGGER_THAN_SIMILARITY = bigger_than_similarity
    THRESHOLD_AVG_LONG = threshold_avg_long
    EXCLUDE_KEYWORDS = exclude_keywords
    
    prog = st.progress(0, text='Ausführen der grundlegenden Datentransformation...')
    data_meta['id'] = data_meta['identifier'].str.extract(r'(\d+)@.*').astype(int)
    data_meta['publisher'] = data_meta.apply(lambda x: x['publisher'][0], axis=1)
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
        # .assign(avg_ratio_color=lambda x: mpl.colors.rgb2hex(_colors(x['avg_ratio_normalized'])))
        .assign(avg_short=lambda x: x['avg_short'].fillna(0))
        .assign(avg_long=lambda x: x['avg_long'].fillna(0))
        .assign(keyword=lambda x: x['keyword'].fillna('').apply(list))
    )
    data_meta['avg_ratio_color'] = data_meta.apply(lambda x: mpl.colors.rgb2hex(_colors(x['avg_ratio_normalized'])), axis=1)

    # Building graph
    logging.info('Build graph...')
    prog.progress(0.2, text='Keywords aller Datensätze werden miteinander verglichen...')
    G = nx.Graph()
    
    # Alternative approach for comparing similarity (much faster than iterrows)
    df_cartesian = data_meta.merge(data_meta, how='cross', suffixes=('_x', '_y'))
    df_cartesian['keyword_x'] = df_cartesian['keyword_x'].apply(set)
    df_cartesian['keyword_y'] = df_cartesian['keyword_y'].apply(set)
    prog.progress(0.4, text='Häufig verwendete Keywords werden gefiltert...')
    df_cartesian['_sub_keyword_x'] = df_cartesian['keyword_x'] - EXCLUDE_KEYWORDS
    df_cartesian['_sub_keyword_y'] = df_cartesian['keyword_y'] - EXCLUDE_KEYWORDS
    df_cartesian['_sub_keyword_x'] = df_cartesian['_sub_keyword_x'].apply(list)
    df_cartesian['_sub_keyword_y'] = df_cartesian['_sub_keyword_y'].apply(list)
    prog.progress(0.5, text='Ähnlichkeit der Datensätze wird berechnet...')
    df_cartesian = df_cartesian[df_cartesian['id_x'] < df_cartesian['id_y']]
    df_cartesian['weight'] = df_cartesian.apply(lambda x: SequenceMatcher(None, x['_sub_keyword_x'], x['_sub_keyword_y']).ratio(), axis=1)
    df_cartesian = df_cartesian[(df_cartesian['weight'] > BIGGER_THAN_SIMILARITY) & (df_cartesian['avg_long_x'] >= THRESHOLD_AVG_LONG) & (df_cartesian['avg_long_y'] >= THRESHOLD_AVG_LONG)]
    prog.progress(0.6, text='Edges werden zum Graph hinzugefügt...')
    df_cartesian.apply(lambda x: G.add_edge(x['id_x'], x['id_y'], weight=x['weight']*2), axis=1)
    logging.info(f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')


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
            f"ID: {row['id']}<br>"
            f"Organisation: {row['publisher']}<br>"
            f"Klicks (letzte {AVG_SHORT_DAYS} Tage): {int(row['avg_short'])}<br>"
            f"Klicks (letzte {AVG_LONG_DAYS} Tage, in Relation zu {AVG_SHORT_DAYS} Tagen): {round(row['avg_long'], 1)}<br>"
            f"Keywords: {', '.join(row['keyword'])}"
        )
    nx.set_node_attributes(G, _hover, 'hover')

    # Relabel nodes, needs to be executed last
    _node_title = data_meta[['id', 'title']].set_index('id').to_dict()['title']
    nx.relabel_nodes(G, _node_title, copy=False)

    logging.info(f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    logging.info('Calculate Graph layout...')
    prog.progress(0.8, text='Simluation des Graphen in 200 iterationen...')
    # pos = nx.spring_layout(G, k=0.04, iterations=200, scale=10_000, seed=42)
    pos = nx.spring_layout(G, iterations=200, scale=scale, seed=42)
    # Add coordinates as node annotations that are recognized by gravis
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
        
    )
    prog.progress(0.95, text='Berechnen Organisationsstatistiken...')
    df_stat = (
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
    
    prog.empty()
    return fig, df_stat


def intro_text(days_short: int = 30):
    return f"""
        Diese Applikation visualisiert die Zugriffsstatistik der publizierenden Organisationen
        des Metadatenkatalogs des Kantons Zürich. Dabei wird ein Graph generiert, der die Ähnlichkeit
        zweier Datensätze darstellt. Verwenden zwei Datensätze ähnliche Keywords, dann stehen sie
        näher beeinander. Die Grösse der Nodes repräsentiert die Anzahl der Zugriffe der letzten `{days_short}` Tage,
        die Farbe des Nodes gibt an, ob die Zugriffe in kürzerer Zeit zugenommen (:red[rot]) oder abgenommen (:blue[blau]) haben.
        Die Schriftfarbe der Nodes gibt die Organisation an, die den Datensatz publiziert hat.
        Quellen:
        * [Metadatenkatalog]({METADATA_URL})
        * [Zugriffs-Statistik]({STATISTICS_URL})
        * [Github-Repo]({GITHUB_URL})
    """
    

logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")
st.title('OGD Kanton Zürich Zugriffsstatistik')
intro = st.markdown(intro_text(30))

input_col1, input_col2 = st.columns(2, gap='medium')
input_col1.subheader('Filter (generell)')
timespan = input_col1.slider(
    'Vergleich Zugriffszahlen (in Tagen)',
    1, 180, (30, 180),
)
intro.write(intro_text(timespan[0]))
input_thresold_avg_long = input_col1.slider(
    'Schwelle für durchschnittliche Zugriffe (letzte 180 Tage)',
    1, 100, 1,
)

input_col2.subheader('Filter (Berechnung Edges)')
excl_keywords = input_col2.multiselect(
    'Auszuschliessende Keywords (beim Vergleich der Ähnlichkeit von Datensätzen)',
    list(available_keywords),
    list(DEFAULT_EXCLUDE_KEYWORDS),
)
input_bigger_than_similarity = input_col2.slider(
    'Ähnlichkeitsschwelle zweier Datensätze anhand Keywords (0-1)',
    0.0, 1.0, 0.0,
    step=0.01,
)
input_scale = input_col2.slider(
    'Skalierung des Graphen (je grösser, desto mehr Platz)',
    1_000, 30_000, 10_000,
    step=1_000,
)

fig, df_stat_out = main(
    data_stat=data_stat,
    data_meta=data_meta,
    avg_long_days=timespan[1],
    avg_short_days=timespan[0],
    exclude_keywords=set(excl_keywords),
    bigger_than_similarity=input_bigger_than_similarity,
    threshold_avg_long=input_thresold_avg_long,
    scale=input_scale,
)
st.subheader('Graph')
_html = fig.to_html_standalone()
st.components.v1.html(_html, height=1208)
st.subheader('Zugriffsstatistik')
st.markdown(
    f'Die nachfolgende Tabelle zeigt die Zugriffsstatistik der publizierenden Organisationen der letzten'
    f'`{timespan[0]}` Tage unter Berücksichtigung der gesetzten Filter.'
)
st.dataframe(
    df_stat_out.style.applymap(lambda col: f"background-color: {col}", subset=['Farbe']),
    hide_index=True
)
