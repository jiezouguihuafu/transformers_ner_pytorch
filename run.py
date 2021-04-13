from transformers import BertConfig,BertTokenizer
from model import get_labels,Trainer,BertCrfForNer,Arguments


def run():
    args = Arguments()
    # get data label
    labels = get_labels(args.labels)
    num_labels = len(labels)

    config = BertConfig.from_pretrained(args.model_name_or_path,num_labels=num_labels,)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertCrfForNer.from_pretrained(args.model_name_or_path,config=config)

    # Training
    trainer = Trainer(args=args,model=model,tokenizer=tokenizer,labels=labels)
    trainer.train()


if __name__ == "__main__":
    run()
