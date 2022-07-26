from transformers import AdamW, get_cosine_schedule_with_warmup

def get_optimizer(optim: str, model, lr, warm_up_steps, training_steps):
    if optim == 'AdamW':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                                    {
                                        'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                        'weight_decay' : 0.01
                                    },
                                    {
                                        'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                        'weight_decay' : 0.0
                                    }
    ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, training_steps)
        return (optimizer, scheduler)