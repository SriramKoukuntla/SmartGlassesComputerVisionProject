On top of downloading dependencies in your virtual environment, you need to download the GPU version of the paddleOcr library. This project won't be viable without it. Without gpu support there is a 1000x slow down in performance.

You can find your specific version of the paddle ocr based on your Nvidia drivers here: https://www.paddlepaddle.org.cn/en. 

For example, the command it gave me to install mine is this:
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/