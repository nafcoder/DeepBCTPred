import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def display_images(images, num_images=10):
    # Limit the number of images displayed if there are too many
    images_to_display = images[:num_images]
    plt.figure(figsize=(15, 5))
    
    for idx, img in enumerate(images_to_display):
        img = (img * np.array([0.229, 0.224, 0.225])[:, None, None]) + np.array([0.485, 0.456, 0.406])[:, None, None]
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8
        plt.subplot(1, num_images, idx + 1)
        
        # Convert image from (C, H, W) to (H, W, C) if needed
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)  # Convert to (H, W, C)
        elif isinstance(img, np.ndarray) and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)

        plt.imshow(img)
        plt.axis('off')
    
    plt.show()


# Define augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomAffine(30, shear=10),
])

# Add noise function
def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 0.1, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)

# Augmented dataset function
def augment_dataset(df, img_dir, transform=None, num_new_images=5):
    augmented_images = []
    labels = []
    
    for _, row in df.iterrows():
        img_name = os.path.join(img_dir, row[1], row[0])
        image = Image.open(img_name).convert("RGB")
        
        # Apply augmentation
        for _ in range(num_new_images):
            augmented_image = image
            augmented_image = augmentation_transforms(augmented_image)
            augmented_image = add_noise(augmented_image)
            
            if transform:
                augmented_image = transform(augmented_image)
                
            augmented_images.append(augmented_image)
            labels.append(row['encoded_label'])
    
    return augmented_images, labels


def evaluate_model(images, labels, model):
    # Flatten each image and convert to a format suitable for LightGBM
    images_flat = np.array([np.array(img).flatten() for img in images])
    labels = np.array(labels)
    
    # Generate predictions using LightGBM
    preds = model.predict(images_flat)
    
    # Calculate accuracy as fitness score
    accuracy = accuracy_score(labels, preds)
    
    return accuracy


def genetic_algorithm_selection(augmented_images, labels, model, num_generations=20, sample_inside_each_individual=500, num_individuals=100, num_top_individuals=70):
    # Initialize the first generation randomly
    population = [random.sample(range(len(augmented_images)), sample_inside_each_individual) for _ in range(num_individuals)]

    for _ in range(num_generations):
        fitness_scores = []
        
        # Evaluate fitness of each individual in the population
        for individual in population:
            images = [augmented_images[i] for i in individual]
            labels_batch = [labels[i] for i in individual]
            accuracy = evaluate_model(images, labels_batch, model)
            fitness_scores.append(accuracy)
        
        # Sort individuals based on fitness scores and select the top individuals
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        top_individuals = sorted_population[:num_top_individuals]
        
        # Generate the next generation by keeping top individuals and mutating them
        next_generation = top_individuals[:]
        while len(next_generation) < num_individuals:
            parent = random.choice(top_individuals)
            child = mutate(parent, len(augmented_images))
            next_generation.append(child)
        
        # Update the population with the next generation
        population = next_generation

    # Return the top individual from the final generation
    top_images = [augmented_images[i] for i in top_individuals[0]]
    top_labels = [labels[i] for i in top_individuals[0]]
    
    return top_images, top_labels


def mutate(individual, image_pool_size, mutation_rate=0.1):
    # Define a simple mutation function to modify an individual slightly
    child = individual[:]
    number_of_mutations = int(mutation_rate * len(individual))
    for _ in range(number_of_mutations):
        idx_to_mutate = random.randint(0, len(individual) - 1)
        child[idx_to_mutate] = random.randint(0, image_pool_size - 1)
    return child


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Load the annotation CSV
train_df = pd.read_csv("train_df.csv")
val_df = pd.read_csv("val_df.csv")

# merge train and val
train_df = pd.concat([train_df, val_df])

# Image transformations for resizing and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# dataset_path = "baldder_tissue_classification"

# augmented_images, augmented_labels = augment_dataset(train_df, dataset_path, transform=transform, num_new_images=5)
# print(train_df.shape)
# print(len(augmented_images))

# # load model from pickle
# model = pickle.load(open("LightGBM.pkl", "rb"))
# # # Apply genetic algorithm to select top images
# top_images, top_labels = genetic_algorithm_selection(augmented_images, augmented_labels, model)

# print(len(top_images))

# top_images_array = np.array(top_images)
# top_labels_array = np.array(top_labels)

# # Save the arrays as .npy files
# np.save("top_images.npy", top_images_array)
# np.save("top_labels.npy", top_labels_array)
loaded_images = np.load("top_images.npy")
loaded_labels = np.load("top_labels.npy")

display_images(loaded_images)