import torch
import sys
import os

# Add the directory containing train_transformer.py to Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from train_transformer import DecoderTransformer, encode, decode, get_vocab
except ImportError as e:
    print(f"Error importing from train_transformer: {e}")
    print("Make sure train_transformer.py is in the same directory and contains the required classes/functions")
    sys.exit(1)

# Use the imported functions in your tests
stoi, itos = get_vocab()
# Example usage:
# tokens = encode("some text", stoi)
# text = decode(tokens, itos)

# Only pass vocab_size since other parameters are constants in the class
model_args = {
    'vocab_size': len(get_vocab()[0])
}

def predict_next_words(input_text, model_path='best_model.pt', num_words=3, max_new_tokens=50):
    """
    Predict the next few words given an input text.
    Args:
        input_text (str): The input text to continue from
        model_path (str): Path to the saved model
        num_words (int): Number of words to predict
        max_new_tokens (int): Maximum number of tokens to generate
    Returns:
        str: Original text with predicted continuation
    """
    # Set up device
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)

    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    # Initialize model with the correct parameters
    model = DecoderTransformer(**model_args)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)  # Use strict loading to catch mismatches
    model.to(device)
    model.eval()

    # Encode the input text
    context = torch.tensor(encode(input_text, stoi), dtype=torch.long, device=device)
    context = context.unsqueeze(0)  # Add batch dimension

    # Generate predictions
    with torch.no_grad():
        output_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0]
        predicted_text = decode(output_tokens.tolist(), itos)

    # Split the text into words and get the next num_words
    all_words = predicted_text.split()
    input_words = input_text.split()
    
    # Find where input text ends and prediction begins
    new_words = all_words[len(input_words):len(input_words) + num_words]
    new_words = ' '.join(new_words)

    return f"Input: {input_text}\nPredicted continuation: {new_words}"

def main():
    """
    Main function to test the model with both predefined examples and user input.
    """
    # Test with predefined examples
    print("Testing with predefined examples:")
    test_texts = [
        "The quick brown fox",
        "In the beginning",
        "She opened the door and",
        "The weather today is",
        "I love to program because"
    ]
    
    print("\n=== Predefined Examples ===")
    for text in test_texts:
        prediction = predict_next_words(text)
        print(prediction)
        print("-" * 50)
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter your text to get predictions (type 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        
        if not user_input:
            print("Please enter some text!")
            continue
        
        try:
            prediction = predict_next_words(
                user_input,
                model_path='best_model.pt',
                num_words=3,
                max_new_tokens=50
            )
            print(prediction)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 