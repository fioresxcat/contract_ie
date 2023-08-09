from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, LayoutLMv3Config, LiltForTokenClassification, LayoutLMv2FeatureExtractor, LayoutXLMTokenizerFast, LayoutXLMProcessor,  LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast


def get_model_from_config(config, label2id, id2label):
    pretrained_path = config.model.pretrained_path
    if config.model.model_type == 'layoutlmv3':
        model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained_path, label2id=label2id, id2label=id2label)
        model.config.label2id = label2id
        model.config.id2label = id2label

        feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(pretrained_path, apply_ocr=False)
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(pretrained_path)
        tokenizer.only_label_first_subword = False
        processor = LayoutLMv3Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        # processor = LayoutLMv3Processor.from_pretrained(pretrained_path, apply_ocr = False)
        # processor.tokenizer.only_label_first_subword = False
    elif config.model.model_type == 'lilt':
        model = LiltForTokenClassification.from_pretrained(pretrained_path, label2id=label2id, id2label=id2label)
        model.config.label2id = label2id
        model.config.id2label = id2label

        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(pretrained_path)
        tokenizer.only_label_first_subword = False
        processor = LayoutXLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    return model, processor