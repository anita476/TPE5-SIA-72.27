import numpy as np
from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from utils.font_loader import load_font,print_all

def main():

    X, bitmaps, labels = load_font("../data/font.h")

    print(f"X shape:       {X.shape}   ← training matrix")
    print(f"bitmaps shape: {bitmaps.shape}")
    print(f"n_chars:       {len(labels)}\n")


    model = SimpleAutoencoder([35, 16, 8, 16, 35], activation="relu", seed=42)
    model.train(X, epochs=50, lr=1e-3, batch_size=64, log_every=10)

    sample = X[:5]
    latent = model.encode(sample)
    reconstructed = model.decode(latent)
    print(f"\nOriginal shape:      {sample.shape}")
    print(f"Latent shape:        {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")


    print("Original letters:")
    print("=======================")

    print_all(X, labels)

    print("Reconstructed letters:")
    print("=======================")
    print(reconstructed)

    print_all(reconstructed, labels)




if __name__ == "__main__":
    main()