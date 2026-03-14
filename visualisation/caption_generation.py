import torch


def generate_caption(model, image, tokenizer, max_len=30):

    model.eval()

    tokens = [tokenizer.bos_token_id]

    for _ in range(max_len):

        input_ids = torch.tensor(tokens).unsqueeze(0).to(image.device)

        outputs = model.text_decoder(input_ids, image)

        next_token = outputs[:,-1].argmax(-1).item()

        tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    caption = tokenizer.decode(tokens)

    return caption