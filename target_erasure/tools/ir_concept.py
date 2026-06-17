# coding: UTF-8
"""
    @date:  2024.12.13  week50 
    @func:  process
"""
import os
import random
import json
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast

class UniversalModelCaller:
    def __init__(self, api_keys):
        """
        """
        self.api_keys = api_keys
        self.model_clients = {}
        self.setup_clients()

    def setup_clients(self):
        """
        """
        if "gpt" in self.api_keys and self.api_keys["gpt"]:
            self.model_clients["gpt"] = {}
            if self.api_keys["gpt"]["azure"]:
                self.model_clients["gpt"]["client"] = AzureOpenAI(
                                                api_key=self.api_keys["gpt"]["api_key"],
                                                api_version=self.api_keys["gpt"]["api_version"],
                                                azure_endpoint=self.api_keys["gpt"]["end_point"])
                self.model_clients["gpt"]["model_name"] = "gpt-4o"  #
            else:
                self.model_clients["gpt"]["client"] = OpenAI(api_key=self.api_keys["gpt"])
                self.model_clients["gpt"]["model_name"] = "gpt-4o"  # 

        if "claude" in self.api_keys and self.api_keys["claude"]:
            self.model_clients["claude"] = {}
            self.model_clients["claude"]["client"] = anthropic.Anthropic(api_key=self.api_keys["claude"])
            self.model_clients["claude"]["model_name"] = "claude-3-5-sonnet-20241022"

        if "kimi" in self.api_keys and self.api_keys["kimi"]:
            self.model_clients["kimi"] = {}
            self.model_clients["kimi"]["client"] = OpenAI(
                api_key=self.api_keys["kimi"], base_url="https://api.moonshot.cn/v1"
            )
            self.model_clients["kimi"]["model_name"] = "moonshot-v1-8k"

        if "qwen" in self.api_keys and self.api_keys["qwen"]:
            self.model_clients["qwen"] = {}
            self.model_clients["qwen"]["client"] = OpenAI(
                api_key=self.api_keys["qwen"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model_clients["qwen"]["model_name"] = "qwen2-72b-instruct"

    def get_response(self, model_name, messages):
        """
        """
        if model_name not in self.model_clients:
            raise ValueError(f"Model {model_name} is not supported or not properly configured.")

        client = self.model_clients[model_name]
        if model_name == "gpt":
            response = client["client"].chat.completions.create(
                model=client["model_name"],
                messages=messages
            )
            return response.choices[0].message.content

        elif model_name == "claude":
            response = client["client"].messages.create(
                model=client["model_name"],
                max_tokens=1024,
                messages=messages
            )
            return response.content

        elif model_name == "kimi":
            response = client["client"].chat.completions.create(
                model=client["model_name"],
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content

        elif model_name == "qwen":
            response = client["client"].chat.completions.create(
                model=client["model_name"],
                messages=messages
            )
            # import pdb;pdb.set_trace()
            return response.choices[0].message.content

        return None


def filt_response(response: str):
    # import pdb;pdb.set_trace()
    response = response.split("{", 1)[1]
    response = response.rsplit("}", )[0]
    response = eval("{" + response.replace("\n", "") + "}")
    # response = json.loads("{" + response + "}")
    return response


def tidy_output(final_output):

    final_tmp = final_output[0]
    idx = random.choice([0, 1, 2])
    
    return [final_tmp['No_relation'][idx][0], final_tmp['far'][idx][0], final_tmp['mid'][idx][0]]


def MoE(model_caller, word, K, model_list=["gpt", "claude", "kimi", "qwen"], final_expert=["gpt"]):
    messages = [{"role": "system", "content": "You are a helpful assistant and a well-established language expert"},
                {"role": "user", "content": 'Hello, please return ' + str(K) + 'English words that you think with Human intuition are far/mid/close in the semantic space from the English word: {'+ word 
                                            + '}, and only reply the result with JSON format is as follows: {"No_relation": [(word1, similarity_score1), ...],"far": [(word1, similarity_score1), ...],"mid": [(word1, similarity_score1),\
                                            ...],"close": [(word1, similarity_score1), ...]}',
            }]
    results = {}
    final_results = []

    if "gpt" in model_list:
        try:
            gpt_response = model_caller.get_response("gpt", messages)
            print("GPT Response:", gpt_response)
            gpt_response = filt_response(gpt_response)
            results["gpt"] = gpt_response
        except Exception as e:
            print(e)
            # import pdb;pdb.set_trace()

    if "claude" in model_list:
        claude_response = model_caller.get_response("claude", messages)
        print("Claude Response:", claude_response)
        claude_response = filt_response(claude_response)
        results["claude"] = claude_response
        
    if "kimi" in model_list:
        kimi_response = model_caller.get_response("kimi", messages)
        print("Kimi Response:", kimi_response)
        kimi_response = filt_response(kimi_response)
        results["kimi"] = kimi_response
    
    if "qwen" in model_list:
        qwen_response = model_caller.get_response("qwen", messages)
        print("Qwen Response:", qwen_response)
        # import pdb;pdb.set_trace()
        qwen_response = filt_response(qwen_response)
        results["Qwen"] = qwen_response
    
    str_results = str(results)

    messages2 =  [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": 'We are currently analyzing' + str(K) + ' words that have different semantic similarities from the word ' + word
                        + ". The experts have provided their results like these:"  + str_results + ". As an expert, please make the final decision on their answer, and just only return the results in the following JSON format with" + str(K) 
                    + ' words:{"No_relation": [(word1, similarity_score1), ...],"far": [(word1, similarity_score1), ...],"mid": [(word1, similarity_score1),\
                                                ...],"close": [(word1, similarity_score1), ...]}',
            }]
    
    if "gpt" in final_expert:
        try:
            gpt_results = model_caller.get_response("gpt", messages2)
            gpt_results = filt_response(gpt_results)
            print("GPT results:", gpt_results)
            final_results.append(gpt_results)
        except Exception as e:
            print(e)
            # import pdb;pdb.set_trace()
    if "qwen" in final_expert:
        try:
            qwen_results = model_caller.get_response("qwen", messages2)
            qwen_results = filt_response(qwen_results)
            print("Qwen results:", qwen_results)
            final_results.append(qwen_results)
        except Exception as e:
            print(e)
         
    irrelevant_strs = tidy_output(final_results)
    return irrelevant_strs

# 기존 API 호출 대신 사용할 로컬 함수
def qwen3_response(model, tokenizer, messages, max_new_tokens=128):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"[SYSTEM]: {content}\n"
        elif role == "user":
            prompt += f"[USER]: {content}\n"
        elif role == "assistant":
            prompt += f"[ASSISTANT]: {content}\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# MoE 함수 수정
def MoE_local_qwen(word, K, device, model_list=["qwen"], final_expert=["qwen"]):
    messages = [{"role": "system", "content": "You are a helpful assistant and a well-established language expert"},
                {"role": "user", "content": 'Hello, please return ' + str(K) + 'English words for each category that you think with Human intuition are far/mid/close in the semantic space from the English word: {'+ word 
                                            + '}, and only reply the result with JSON format is as follows: {"No_relation": [(word1, similarity_score1), ...],"far": [(word1, similarity_score1), ...],"mid": [(word1, similarity_score1),\
                                            ...],"close": [(word1, similarity_score1), ...]}',
            }]

    results = {}
    final_results = []

    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True
    )

    # 프롬프트 구성
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
    response = response.split("</think>")[1]
    ast.literal_eval(response)
    print(response)

    # response = filt_response(response)
    results['qwen'] = response

    # if "qwen" in model_list:
    #     try:
    #         qwen_response = qwen3_response(model, tokenizer, messages)
    #         print("Qwen Response:", qwen_response)
    #         qwen_response = filt_response(qwen_response)  # 기존 함수 사용
    #         results["qwen"] = qwen_response
    #     except Exception as e:
    #         print(e)

    str_results = str(results)

    messages2 =  [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": 'We are currently analyzing' + str(K) + ' words that have different semantic similarities from the word ' + word
                        + ". The experts have provided their results like these:"  + str_results + ". As an expert, please make the final decision on their answer, and just only return the results in the following JSON format with" + str(K) 
                    + ' words:{"No_relation": [(word1, similarity_score1), ...],"far": [(word1, similarity_score1), ...],"mid": [(word1, similarity_score1),\
                                                ...],"close": [(word1, similarity_score1), ...]}',
            }]
    
    text2 = tokenizer.apply_chat_template(
        messages2,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs2 = tokenizer([text2], return_tensors="pt").to(model.device)
    outputs2 = model.generate(
        **model_inputs2,
        max_new_tokens=32768
    )
    response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

    response2 = response2.split("</think>")[1]
    response2 = ast.literal_eval(response2)
    print(response2)

    # response2 = filt_response(response2)
    final_results.append(response2)
    
    # if "qwen" in final_expert:
    #     try:
    #         qwen_results = qwen3_response(messages2)
    #         qwen_results = filt_response(qwen_results)
    #         print("Final Qwen results:", qwen_results)
    #         final_results.append(qwen_results)
    #     except Exception as e:
    #         print(e)
    
    # ===============================
    # GPU 메모리 정리
    # ===============================
    del model
    del model_inputs
    del model_inputs2
    del tokenizer
    torch.cuda.empty_cache()
    # ===============================
    
    irrelevant_strs = tidy_output(final_results)
    return irrelevant_strs

# =====================================================================
# Superclass Extraction via Qwen3-30B Using MoE-like Hierarchical Reasoning
# =====================================================================
def MoE_superclass_qwen(word, K, device="cuda"):

    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

    # -------------------------------
    # Load tokenizer/model
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    # =====================================================================
    # Step 1: Expert Generation (Qwen expert)
    # =====================================================================
    messages = [
        {
            "role": "system",
            "content": "You are an expert linguist specialized in semantic hierarchy and taxonomic reasoning."
        },
        {
            "role": "user",
            "content": f"""
        Given the English word '{word}', list exactly {K} candidate superclasses at **three abstraction levels**:

        1. fine_superclass   : the immediate parent concepts (fine-grained hypernyms)
        2. mid_superclass    : general categories (mid-level hypernyms)
        3. coarse_superclass : very broad semantic categories (coarse-level hypernyms)

        Return ONLY the following JSON format:

        {{
        'fine_superclass': [(super1, score1), ...],
        'mid_superclass': [(super1, score1), ...],
        'coarse_superclass': [(super1, score1), ...]
        }}
        """
                }
            ]

    # ---- prepare text ----
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # ---- generate ----
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.2,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---- remove <think> section ----
    if "</think>" in response:
        response = response.split("</think>")[1]
    # ---- parse JSON ----
    try:
        expert_result = ast.literal_eval(response)
    except:
        print("Parsing failed, raw output:")
        print(response)
        raise

    # store expert result
    experts = {"qwen": expert_result}


    # =====================================================================
    # Step 2: Final Aggregation (meta-expert)
    # =====================================================================
    messages2 = [
        {"role": "system", "content": "You are an expert semantic classifier."},
        {
            "role": "user",
            "content": f"""
        We asked several experts to determine superclasses of the English word '{word}'.
        Here are their responses:

        {str(experts)}

        Please make a final unified decision by selecting the most semantically appropriate superclass candidates.

        Return ONLY JSON in the following format (exactly {K} items for each level):

        {{
        'fine_superclass': [(super1, score1), ...],
        'mid_superclass': [(super1, score1), ...],
        'coarse_superclass': [(super1, score1), ...]
        }}
        """
                }
            ]

    text2 = tokenizer.apply_chat_template(
        messages2,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs2 = tokenizer([text2], return_tensors="pt").to(model.device)

    outputs2 = model.generate(
        **model_inputs2,
        max_new_tokens=32768,
        temperature=0.2
    )
    response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

    if "</think>" in response2:
        response2 = response2.split("</think>")[1]

    final_decision = ast.literal_eval(response2)

    # -------------------------------
    # Cleanup GPU memory
    # -------------------------------
    del model
    del tokenizer
    torch.cuda.empty_cache()

    final_concept = final_decision['fine_superclass']

    return final_concept

    
    
if __name__ == "__main__":
    API_KEY='xxx'
    END_POINT='https://research-01-02.openai.azure.com/'
    API_VERSION = "2024-08-01-preview"

    api_keys = {
        "gpt": {"azure":True, "api_key":API_KEY, "end_point":END_POINT, "api_version": API_VERSION},
        "claude": None,
        "kimi": None,
        "qwen": None
    }

    model_caller = UniversalModelCaller(api_keys)
    irrelevant_concepts = MoE(model_caller, "pablo picasso", 3, model_list=["gpt"])