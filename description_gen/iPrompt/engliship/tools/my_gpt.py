from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import torch
import openai
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel

temp=0.7
max_tokens = 1024
top_p=1.0
freq_penalty=0.4
pres_penalty=0.4
stop_seq=['###']

def get_persona():
    fewshot_prompt = "### User's persona: \n" \
                     "1. Personality Trait | personality trait,\n" \
                     "2. Sport | sport,\n" \
                     "3. Media Genre | media genre. \n" \
                     "User's big five personality: (0.113, 0.857, 0.057, 0.86, 0.711)\n" \
                     "The Big Five personality vector describes the degree of each trait, with a value between 0 and 1: Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism. \n" \
                     "Openness: Closer to 1 means the user is more creative or curious, closer to 0 means the user is more cautious;\n" \
                     "Conscientiousness: Closer to 1 means the user is more productive or organized, and a closer to 0 means the user is more extravagant or careless;\n" \
                     "Extraversion: Closer to 1 means the user is more extroverted or energetic, and closer to 0 means the user is more lonely or introverted;\n" \
                     "Agreeableness: Closer to 1 means the user is more friendly or compassionate, and closer to 0 means that the user is more critical or rational;\n" \
                     "Neuroticism: Closer to 1 means the user is more sensitive or nervous, and closer to 0 means the user is more confident\n\n" \
                     "Generate six profile sentences related to the given user's persona, personaliity and the \"personality trait, sport, media genre\" in each sentence and a description about the user's personality:\n"
    fewshot_prompt_l = "1. I'm an introvert who loves spending time alone. (personality trait: introvert)\n" \
                       "2. I always enjoy running alone on the road at night. (sport: running)\n" \
                       "3. I'm an independent thinker who likes to go against the grain. (personality trait: independent thinker)\n" \
                       "4. I'm a highly sensitive person who feels things deeply. (personality trait: highly sensitive)\n" \
                       "5. I'm an empathetic person who feels connected to others' emotions.(personality trait: empathetic)\n" \
                       "6. I enjoy watching Shakespeare's tragedies very much.(media genre: tragedy)\n" \
                       "The description of (0.113, 0.857, 0.057, 0.86, 0.711): The user is introverted, likes to be alone, has serious obsessive-compulsive disorder and control desire, strong personality, sensitive, weak emotional control ability but very responsible, dedicated to friends. \n\n"

    target_prompt = "### User's persona: \n" \
                    "1. Activity | activity,\n" \
                    "2. Hobby | hobby,\n" \
                    "3. Food | food. \n" \
                    "User's big five personality: (0.974, 0.014, 0.897, 0.145, 0.955)\n" \
                    "The Big Five personality vector describes the degree of each trait, with a value between 0 and 1: Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism. \n" \
                    "Openness: Closer to 1 means the user is more creative or curious, closer to 0 means the user is more cautious;\n" \
                    "Conscientiousness: Closer to 1 means the user is more productive or organized, and a closer to 0 means the user is more extravagant or careless;\n" \
                    "Extraversion: Closer to 1 means the user is more extroverted or energetic, and closer to 0 means the user is more lonely or introverted;\n" \
                    "Agreeableness: Closer to 1 means the user is more friendly or compassionate, and closer to 0 means that the user is more critical or rational;\n" \
                    "Neuroticism: Closer to 1 means the user is more sensitive or nervous, and closer to 0 means the user is more confident\n\n" \
                    "Generate six profile sentences related to the given user's persona, personality and the \"activity, hobby, food\" in each sentence and three descriptions about the user's personality:\n"
    prompt_input = fewshot_prompt + fewshot_prompt_l \
                   + target_prompt

    # print(prompt_input)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt_input,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=freq_penalty,
        presence_penalty=pres_penalty,
        stop=stop_seq,
        # n=1,
        # logprobs=5,
    )
    res = response['choices'][0]['text']
    res_l = res.split('\n')
    res_l_n = extract_person(res_l)
    print(res_l_n)
    return res_l_n

def extract_person(persona_l):
    start_index = 2
    persona_l_e = list()
    for string in persona_l[:5]:
        end_index = string.find("(")
        if end_index != -1:  # 如果找到了 "("
            result = string[start_index:end_index]  # 提取字段
        else:
            result = string[start_index:]  # 如果没有找到 "("，则提取到字符串末尾
        persona_l_e.append(result.strip())
    return persona_l_e

def pad_batch(tokens, pad_id):
    # context_lengths = []
    seq_length = 200
    context_length = len(tokens)
    if context_length < seq_length:
        tokens.extend([pad_id] * (seq_length - context_length))
    return tokens, context_length


def calculate_perplexity(model, tokenizer, input, output):
    # raw_text_len = len(input)
    context_tokens = tokenizer.tokenize(input)
    output_context_tokens = tokenizer.tokenize(output)
    # context_tokens_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(context_tokens))
    # output_tokens_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(output_context_tokens))

    output_length = len(output_context_tokens)
    context_length = len(context_tokens)


    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eos_token_id)
    context_tokens_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(context_tokens))
    # print(args.seq_length)
    # print(context_lengths,context_tokens)
    # context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    # output_tokens_tensor = torch.cuda.LongTensor([output_context_tokens])

    # eos_id = tokenizer.eod
    # tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    # layer_past = None
    counter = 0
    # is_done = torch.zeros([1]).byte().cuda()
    done = False
    score = 0
    print("hii")
    while (counter < output_length and (done == False)):
        # only the recompute runs correctly
        # print(context_tokens_tensor)
        output = model(context_tokens_tensor)
        logits = output[0]
        # print(logits.shape)
        # print(logits)
        # logits = output[:, -1].view(batch_size, -1).contiguous()

        # logits = output[:, context_length - 1, :]
        logits = logits.float()
        log_probs = F.softmax(logits, dim=-1)
        # print(log_probs)
        prev = output_context_tokens[counter]
        log_num = torch.log(log_probs).data
        # print(log_num[0])
        # print(log_num)
        # print(prev)
        # print(log_num[0, counter])
        score += log_num[0, context_length]
        # print(prev)
        context_tokens[context_length] = output_context_tokens[counter]
        context_tokens_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(context_tokens))
        # print(type(context_tokens_tensor))
        context_length += 1
        counter += 1

    return score

def main():
    mode = 0
    # 设置模型配置
    # config = GPT2Config.from_pretrained('gpt2')

    # 创建GPT模型
    # model = GPT2Model(config)
    #
    # # 加载checkpoint的权重参数
    # model.load_state_dict(torch.load('../cp/release/mp_rank_00/model_optim_rng.pt'))
    #
    # # 初始化tokenizer


    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to("cpu")
    input_big = '(0.974, 0.014, 0.897, 0.145, 0.955)'

    persona_des = get_persona()
    persona_score = list()
    # 构造问题
    for persona in persona_des:
        input_str = "\"" + persona + "\"describes the user:\" "
        output_str = input_big + "\""
        score1 = calculate_perplexity(model, tokenizer, input_str, output_str)
        persona_score.append(score1)
        print(score1)
    # print("length: %d" % len(persona_score))
    max_index = np.argmax(np.array(persona_score))
    # max_index, max_value = max(enumerate(persona_score), key=lambda x: x[1])
    print("max_index: %s, max_value: %s" % (max_index,persona_des[max_index]))










if __name__ == "__main__":

    main()