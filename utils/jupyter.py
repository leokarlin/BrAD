from ipywidgets import interact_manual
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from IPython.display import HTML, Image
import numpy as np


def _src_from_img_obj(img_obj):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'data:{mimetype};base64,{b64value}'


def _src_from_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = Image(data=data)
    return _src_from_img_obj(img_obj)


def gallery(images, row_height='auto', caption_prefix=''):
    """Shows a set of images in a gallery that flexes with the width of the notebook.

    Parameters
    ----------
    images: list of str or bytes
        URLs or bytes of images to display

    row_height: str
        CSS height value to assign to all images. Set to 'auto' by default to show images
        with their native dimensions. Set to a value like '250px' to make all rows
        in the gallery equal height.
    """
    figures = []
    for image in images:
        if isinstance(image, tuple):
            caption_prefix = image[1]
            image = image[0]
        if isinstance(image, bytes):
            src = _src_from_data(image)
            caption = caption_prefix
        elif isinstance(image, Image):
            src = _src_from_img_obj(image)
            caption = caption_prefix
        elif isinstance(image,str):
            src = _src_from_img_obj(Image(filename=image))
            caption = caption_prefix
        else:
            src = image
            caption = f'<figcaption style="font-size: 0.6em">{caption_prefix}{image}</figcaption>'
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

# seek neighbors of an image


def demo(query_domains, search_domains, data_root):

    def show_res(query_domain, search_domain, num_results_per_domain=5):
        src_data_dict = {}
        if search_domain == 'all':
            for d_name, file_path in search_domains.items():
                with open(file_path, 'rb') as fp:
                    src_data = pickle.load(fp)
                    src_nn_fit = NearestNeighbors(n_neighbors=num_results_per_domain, algorithm='auto', n_jobs=-1).fit(
                        src_data[1])
                    src_data_dict[d_name] = (src_data, src_nn_fit)
        else:

            with open(search_domains[search_domain], 'rb') as fp:
                src_data = pickle.load(fp)
                src_nn_fit = NearestNeighbors(n_neighbors=num_results_per_domain, algorithm='auto', n_jobs=-1).fit(
                    src_data[1])
                src_data_dict[search_domain] = (src_data, src_nn_fit)

        with open(query_domains[query_domain], 'rb') as fp:
            dst_data = pickle.load(fp)

        @interact_manual(dst_idx=(0, len(dst_data[0]) - 2))
        def query(dst_idx):
            dst_img_path = os.path.join(data_root, dst_data[0][dst_idx])
            img_paths = [dst_img_path]
            q_cl = dst_img_path.split('/')[-2]
            captions = [f'Query: {q_cl}']
            print(f'\033[1m Query domain: {query_domain}\033[0m')
            display(gallery(zip(img_paths, captions), row_height='100px'))
            for s_domain, s_data in src_data_dict.items():
                _, top_n_matches_ids = s_data[1].kneighbors(dst_data[1][dst_idx:dst_idx + 1])
                top_n_labels = s_data[0][2][top_n_matches_ids][0]
                src_img_pths = [os.path.join(data_root, s_data[0][0][ix]) for ix in top_n_matches_ids[0]]
                img_paths = src_img_pths
                captions = []

                for p in src_img_pths:
                    src_cl = p.split('/')[-2]
                    src_cl = f'<span style="color: #ff0000"> {src_cl} </span>' if src_cl == q_cl else src_cl
                    src_file = p.split('/')[-1]
                    captions.append(src_cl)
                print(f'\033[1m Domain: {s_domain}\033[0m')
                display(gallery(zip(img_paths, captions), row_height='100px'))
    interactive = interact_manual(show_res, query_domain=list(query_domains.keys()),
                                  search_domain=['all'] + list(search_domains.keys()),
                                  num_results_per_domain=np.arange(10))
    interactive.widget.children[-2].description = 'Select Domains'
    d = display(interactive)