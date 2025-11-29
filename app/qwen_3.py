from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def get_model():
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def answer(model, tokenizer, classification: pd.DataFrame, input_df: pd.DataFrame):

    prompt = """
        Você receberá informações sobre um aplicante a linha de crédito, e a classificação final
        do mesmo, ou seja, se é um bom pagador ou não. Você receberá a classificação dada por 3
        modelos, e você deverá fazer um texto bem curto e resumido apresentando possíveis razões
        para os resultados. Você deverá responder *apenas* em português.
        
        A classificação virá no formato:
        
        <modelo> - <classificação>
        
        Com isso, seguem as classificações para você escrever o texto sobre:
        
    """

    if isinstance(classification, pd.DataFrame):
        if "model" in classification.columns and "prediction" in classification.columns:
            for _, row in classification[["model", "prediction"]].iterrows():
                prompt += f"{row['model']}: {row['prediction']}\n"
        else:
            for col in classification.columns[:2]:
                prompt += f"{col}: {classification[col].astype(str).tolist()}\n"
    else:
        prompt += str(classification)

    prompt += """
        E esses foram os inputs passados pelo usuário, para te ajudar na análise e produção do texto:
        
    """

    prompt += "\nInputs fornecidos pelo usuário:\n"

    for col in input_df.columns:
        val = input_df.iloc[0][col]
        prompt += f"{col}: {val}\n"

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.1
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content
