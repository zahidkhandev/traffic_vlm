from k_means_qc import KMeansQC

data_path = "data/raw"
data_split = "test"
labels_path = data_path + "/labels/" + data_split
images_path = data_path + "/images/" + data_split
sample_file = "/cabc30fc-e7726578"

k_means_obj = KMeansQC()

sample_file_json = labels_path + sample_file + ".json"
sample_file_image = images_path + sample_file + ".jpg"

x = k_means_obj.extract_embeddings(
    label_json_path=sample_file_json, image_path=sample_file_image
)

print(x)

k_means_obj.loop_labels(
    labels_path=labels_path,
    images_path=images_path,
    output_dir="data/processed/k_means_qc",
)
