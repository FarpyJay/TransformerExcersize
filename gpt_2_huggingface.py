import matplotlib.pyplot as plt
from transformers import pipeline, set_seed, GPT2LMHeadModel


def huggingface_response():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    return generator("Hello, I'm a language model", max_length=30, num_return_sequences=5)

def huggingface_model(model_to_use='gpt2'):
    model_hf = GPT2LMHeadModel.from_pretrained(model_to_use)
    state_dict_hf = model_hf.state_dict()
    return state_dict_hf

    # plt.plot(state_dict_hf['transformer.wpe.weight'][:, 150])
    # plt.plot(state_dict_hf['transformer.wpe.weight'][:, 200])
    # plt.plot(state_dict_hf['transformer.wpe.weight'][:, 250])
    # plt.savefig('state_dict_rows')
    # plt.imshow(state_dict_hf['transformer.wpe.weight'], cmap='gray')
    # plt.savefig('wpe_map')
    # for k, v in state_dict_hf.items():
    #     print(k, v.shape)

if __name__ == '__main__':
    huggingface_model()