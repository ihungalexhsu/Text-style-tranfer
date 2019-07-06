## Instructions
This folder provide two method of creating self-attention based word selection
method.
- Multi-Head self-attention
  - attn = softmax((W_q.Q)(W_k.K)) for each time
  - use the last hidden state as query
  - concat context from each time and pass the context to MLP classifier
  - use 2 heads in our implementation
- Structure self-attention (https://arxiv.org/pdf/1703.03130.pdf)
  - attn = softmax(W_s2.tanh(W_s1.K))
  - Ws2 set as 2-by-dimA

## Execution
- Self attention method
  - python main.py -c config/config_2head_selfattn.yaml -m selfatt --load_model --get_align
- Sturcture self-attnetion
  - Not yet developed
