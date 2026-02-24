our-chat
========
LLM from scratch with chat for research and learning. Complete transformer based LLM with support for pretraining and prompt question answer training from files and from online training sets. The code is completely self contained and should be strait forward to experiment with.

Intrinsic math and number "understanding"
-----------------------------------------
One of the main things this approach tries to investigate in teaching an LLM intrinsic understanding of numbers and basic matth.
The idea (and implementation) is that after looking up tokens to get mutidimensional vectors representing each token, a number of elements are used to represent any parsed number in binary form. So some of the elements in each vector representing a token/word
has weights (values) of ether 0 or 1 matching the 2s complement binary representation of the given number. The results from current experiments with a small (thus fast to train) 2M model is quite promising. In that it does seem to learn how to do basic math beyony the training examples.

Currently the code works with signed integers, but a next small step will be to simply used fixed point real numbers.


A different aprach to tokens/words
----------------------------------
Another approach that the current model is not using traditional tokens that typically are word fragments, but rather complete words. However the model represent a word as two IDs: a unique ID for tha main base word (eg. 'become' = 177) and then one modifier ( 5 different in vase of verbs). For example the 5 forms of the word 'become' would then be:

 - become   : 177, 8
 - became   : 177, 9
 - become   : 177, 10
 - becoming : 177, 11
 - becomes  : 177, 12

 And for another verb say 'balance' it would be:

 - balance   : 435, 8
 - balanced  : 435, 9
 - balanced  : 435, 10
 - balancing : 435, 11
 - balances  : 435, 12

Similar for nouns, adjectives etc.
The idea is to better represent meanings directly in the encodiing and hopefully in a way more akin to how we humans do it.


Documentation
=============
Will be added as soon as I have a coherent system, that I believe others can benefit from using. Hopefully no later than Easter 2026. Also not the the code is really "research style" and not as polished as I would normally prefer.  Partly because of the reasearch element as I did not initially know excactly how to make the code for an LLM, but have learned the details as I went. I do clean up a little with every commit though.



References
==========
 - Originally inspired by: https://livebook.manning.com/book/build-a-large-language-model-from-scratch

