import shutil
shutil.copy('./src/models/pipeline_stable_unclip_img2img.py', './SEED/models/pipeline_stable_unclip_img2img.py')
shutil.copy('./src/models/seed_llama_tokenizer.py', './SEED/models/seed_llama_tokenizer.py')

print('Batch Generation Is Enabled for SEED')