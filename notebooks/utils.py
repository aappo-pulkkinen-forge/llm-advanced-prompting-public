import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download {url} - Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def load_image_data(tsv_file, output_folder, n_images: int = 10000):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load TSV file into DataFrame
    df = pd.read_csv(tsv_file, sep="\t")

    # Sort by photo_width
    df_sorted = df.sort_values(by="photo_width", ascending=True)

    # Get the first 1000 image URLs
    top_images = df_sorted.head(n_images)

    # Download images using multiple threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        for index, row in top_images.iterrows():
            image_url = row["photo_image_url"]
            image_name = os.path.join(output_folder, f"image_{index}.jpg")
            executor.submit(download_image, image_url, image_name)

    print("Download complete!")


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def generate_embeddings(image_model, image_folder, pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            embeddings = pickle.load(f)
        print("Loaded embeddings from pickle file.")
        return embeddings
    image_files = [
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith((".jpg", ".png", ".jpeg"))
    ]

    embeddings = {}

    for image_path in tqdm(image_files):
        image = load_image(image_path)
        image_embedding = image_model.encode(image, convert_to_tensor=True)  # <- Näin
        embeddings[image_path] = image_embedding.detach().cpu().numpy()

    # Save embeddings to pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def search_similar(query_embedding, embeddings, top_k=5):
    image_paths = list(embeddings.keys())
    emb_matrix = np.array(list(embeddings.values()))

    similarities = cosine_similarity([query_embedding], emb_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(image_paths[i], similarities[i]) for i in top_indices]


def visualize_images(image_paths, title="Image Results"):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    if len(image_paths) == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path[0]) if isinstance(img_path[0], str) else img_path[0]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(str(img_path[1]))

    plt.suptitle(title)
    plt.show()


def upload_and_search(image_model, embeddings, device):
    uploader = widgets.FileUpload(accept="image/*", multiple=False)
    search_button = widgets.Button(description="Search Similar Images")
    output = widgets.Output()

    display(uploader, search_button, output)

    def on_search_clicked(b):
        with output:
            clear_output(wait=True)
            if uploader.value:
                print(uploader.value)
                uploaded_file = list(uploader.value)[0]
                image = Image.open(io.BytesIO(uploaded_file["content"])).convert("RGB")
                visualize_images([(image, 0)])
                image_embedding = (
                    image_model.encode(image, convert_to_tensor=True)
                    .to(device)
                    .detach()
                    .cpu()
                    .numpy()
                )

                results = search_similar(image_embedding, embeddings)
                print("Top similar images:", results)
                visualize_images(results, title="Top Similar Images")
            else:
                print("Please upload an image before searching.")

    search_button.on_click(on_search_clicked)


def search_with_image(image_model, embeddings, device, image_path):
    sample_image = load_image(image_path)
    visualize_images([(sample_image, 0)])
    image_query_embedding = (
        image_model.encode(sample_image, convert_to_tensor=True)
        .to(device)
        .detach()
        .cpu()
        .numpy()
    )

    results = search_similar(image_query_embedding, embeddings)

    visualize_images(results)


def search_with_text(text_model, text, embeddings, device):
    text_query_embedding = (
        text_model.encode(text, convert_to_tensor=True)
        .to(device)
        .detach()
        .cpu()
        .numpy()
    )

    results = search_similar(text_query_embedding, embeddings)

    visualize_images(results, title=f"Text search: {text}")
