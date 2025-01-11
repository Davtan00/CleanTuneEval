from transformers import AutoModel

def main():
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    print("Model loaded successfully!")

    for name, module in model.named_modules():
        print(name)
if __name__ == '__main__':
    main()