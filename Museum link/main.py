import os
import time
import matplotlib.pyplot as plt
from collections import Counter
from models import resnet_model, vgg16_model, inceptionv3_model



def current_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

image_path="Museum link/images" # Remember to switch this to /images instead of /testdata

print("starting process")

resnet_results = resnet_model(image_path)
vgg16_results = vgg16_model(image_path)
inceptionv3_results = inceptionv3_model(image_path)

print("All images proccessed")

def write_to_txt_file( results_file='results.txt', output_dir="Museum link/results"):
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, results_file)
    
    print(f"Start writing. Writing to directory: {output_dir}")

    with open(results_path, 'w') as file:
        
        timestamp = current_timestamp()

        file.write(f"Timestamp: {timestamp}\n")

        file.write("ResNet50 Results:\n")
        for file_name, results in resnet_results.items():
            file.write(f"{file_name}:\n")
            for category, probability in results:
                file.write(f"  {category}: {probability}\n")

        file.write("\nVGG16 Results:\n")
        for file_name, results in vgg16_results.items():
            file.write(f"{file_name}:\n")
            for category, probability in results:
                file.write(f"  {category}: {probability}\n")

        file.write("\nInceptionV3 Results:\n")
        for file_name, results in inceptionv3_results.items():
            file.write(f"{file_name}:\n")
            for category, probability in results:
                file.write(f"  {category}: {probability}\n")

    print("Done writing")


def generate_pie_charts(output_dir='Museum link/charts'):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating charts. Charts will be in {output_dir}")

    combined_category_counts = Counter()
    
    for model_name, model_results in zip(["resnet50", "vgg16", "inceptionv3"], [resnet_results, vgg16_results, inceptionv3_results]):
        categories = [category for results_list in model_results.values() for category, _ in results_list]
        category_counts = Counter(categories)
        categories = list(category_counts.keys())[:10]
        counts = list(category_counts.values())[:10]

        combined_category_counts.update(category_counts)
        
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=categories, startangle=140, autopct='%1.1f%%')
        plt.title(f'Distribution of Categories ({model_name})')
        plt.axis('equal')
        
        filename = f"{model_name}_chart.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    categories = list(combined_category_counts.keys())[:10]
    counts = list(combined_category_counts.values())[:10]
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=categories, startangle=140, autopct='%1.1f%%')
    plt.title('Combined Distribution of Categories')
    plt.axis('equal')
    
    filename = "combined_chart.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    print("Charts completed")


if __name__ == "__main__":
    
    write_to_txt_file()
    generate_pie_charts()
