def save_income(img, mask, prompt, model, guidance_scale, timestamp):
    mask.save(f"./persistence/img/input/{timestamp}_mask.png")
    img.save(f"./persistence/img/input/{timestamp}_image.png")
    txt = f'timestamp:{timestamp}, prompt:{prompt}, model:{model}, guidance_scale:{guidance_scale}\n'
    with open("./persistence/prompt/prompts.txt", "a", encoding="utf-8") as file:
        file.write(txt)


def save_result(img, timestamp):
    img.save(f"./persistence/img/result/{timestamp}.png")