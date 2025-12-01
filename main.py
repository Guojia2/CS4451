def main():
    print("FreDF Training")
    print("Usage: python main.py --backbone [itransformer|tsmixer]")
    print()
    
    # import and run training
    from trains.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()

