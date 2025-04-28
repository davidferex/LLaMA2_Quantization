# LLaMA2 Quantization
This repository contains the code and resources for my final project in Computer Science.
In this work, I explore and evaluate various quantization techniques applied to large language models (LLMs). Specifically, I focus on the LLaMA 2 7B model as the base model for experimentation.
The objective is to quantify the impact of different quantization methods by comparing the performance, memory usage, and efficiency of the quantized models against the original full-precision model. Specifically, I use BitsAndBytes, GPTQ, HQQ and Quanto as quantization methods.

Its structure is separated by method. For example, in the GPTQ folder you can find all the code used for quantization and testing with the GPTQ method. The only exception is perplexity related code, which has its own folder. The main files are the notebooks, which are available in the main directory, outside any folder. This notebooks contain all the results for each quantization method.

## License

This project is protected by copyright. Unless otherwise stated, copying, modifying, distributing, or using this content without the explicit permission of the author is not allowed.


