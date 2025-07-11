## Main Performance

### Average loss training

**Accuracy**

| model | T | E | M | H | A |
| ----- | - | - | - | - | - |
| dense | A | 0.7067 | 0.5446 | 0.3961 | 0.5589 |
| mlp   | A | 0.6918 | 0.5461 | 0.3961 | 0.5539 |
| rnn   | A | 0.7054 | 0.5581 | 0.3917 | 0.5615 |
| r-m   | A | 0.6900 | 0.5567 | 0.3969 | 0.5569 |
| dense | P | 0.6775 | 0.5319 | 0.3968 | 0.5443 |
| mlp   | P | 0.6825 | 0.5468 | 0.3946 | 0.5502 |
| rnn   | P | 0.6986 | 0.5426 | 0.3931 | 0.5544 |
| r-m   | P | 0.6770 | 0.5298 | 0.3850 | 0.5397 |

**PSNR(dB)**

| model | T | E | M | H | A |
| ----- | - | - | - | - | - |
| dense | A | 22.8618 | 19.3108 | 20.6649 | 21.0405 |
| mlp   | A | 23.6942 | 19.9731 | 20.6325 | 21.5522 |
| rnn   | A | 23.4181 | 19.4942 | 20.7607 | 21.3350 |
| r-m   | A | 22.8932 | 19.6607 | 20.5675 | 21.1351 |
| dense | P | 23.3965 | 19.6468 | 20.7225 | 21.3645 |
| mlp   | P | 23.9818 | 19.9239 | 20.9256 | 21.7328 |
| rnn   | P | 23.4132 | 19.6108 | 20.7928 | 21.3807 |
| r-m   | P | 23.6587 | 19.7822 | 20.7671 | 21.5189 |

**SSIM**

| model | T | E | M | H | A |
| ----- | - | - | - | - | - |
| dense | A | 0.8078 | 0.5359 | 0.6372 | 0.6677 |
| mlp   | A | 0.7963 | 0.5318 | 0.6195 | 0.6566 |
| rnn   | A | 0.8100 | 0.5339 | 0.6369 | 0.6676 |
| r-m   | A | 0.7866 | 0.5272 | 0.6122 | 0.6493 |
| dense | P | 0.8004 | 0.5333 | 0.6289 | 0.6615 |
| mlp   | P | 0.8101 | 0.5385 | 0.6370 | 0.6692 |
| rnn   | P | 0.8086 | 0.5362 | 0.6374 | 0.6680 |
| r-m   | P | 0.8029 | 0.5336 | 0.6297 | 0.6628 |

### Uncertainty-Weighted loss training

**Accuracy**

| model | T | E | M | H | A |
| ----- | - | - | - | - | - |
| dense | A | 0.7197 | 0.5532 | 0.4028 | 0.5686 |
| mlp   | A | 0.6906 | 0.5418 | 0.3939 | 0.5514 |
| rnn   | A | 0.7110 | 0.5667 | 0.3931 | 0.5667 |
| r-m   | A | 0.7172 | 0.5447 | 0.3887 | 0.5605 |
| dense | T | 0.7079 | 0.5532 | 0.3879 | 0.5596 |
| mlp   | P | 0.6906 | 0.5418 | 0.3939 | 0.5514 |
| rnn   | P | 0.7004 | 0.5631 | 0.3842 | 0.5589 |
| r-m   | P | 0.7172 | 0.5447 | 0.3887 | 0.5605 |

**PSNR(dB)**

| model | T | E | M | H | A |
| ----- | - | - | - | - | - |
| dense | A | 23.5562 | 19.5168 | 20.7582 | 21.3926 |
| mlp   | A | 23.6682 | 19.8244 | 20.8388 | 21.5580 |
| rnn   | A | 23.0507 | 19.6382 | 20.7841 | 21.2528 |
| r-m   | A | 24.0507 | 20.0122 | 20.8662 | 21.7686 |
| dense | P | 23.4966 | 19.7010 | 20.7817 | 21.4372 |
| mlp   | P | 23.6682 | 19.8244 | 20.8388 | 21.5580 |
| rnn   | P | 23.5369 | 19.8643 | 20.7437 | 21.4931 |
| r-m   | P | 24.0507 | 20.0122 | 20.8662 | 21.7686 |

**SSIM**

| model | T | E | M | H | A |
| ----- | - | - | - | - | - |
| dense | A | 0.8159 | 0.5390 | 0.6431 | 0.6734 |
| mlp   | A | 0.7980 | 0.5350 | 0.6299 | 0.6615 |
| rnn   | A | 0.8043 | 0.5359 | 0.6351 | 0.6657 |
| r-m   | A | 0.8049 | 0.5377 | 0.6334 | 0.6660 |
| dense | P | 0.8047 | 0.5382 | 0.6322 | 0.6657 |
| mlp   | P | 0.7980 | 0.5350 | 0.6299 | 0.6615 |
| rnn   | P | 0.8061 | 0.5362 | 0.6328 | 0.6657 |
| r-m   | P | 0.8049 | 0.5377 | 0.6334 | 0.6660 |

### Number of SRB

**Accuracy**

| N | T | E | M | H | A |
| - | - | - | - | - | - |
| 0 | A | 0.6770 | 0.5113 | 0.3745 | 0.5306 |
| 1 | A | 0.6795 | 0.5411 | 0.3857 | 0.5445 |
| 2 | A | 0.6702 | 0.5468 | 0.3753 | 0.5397 |
| 3 | A | 0.7172 | 0.5447 | 0.3887 | 0.5605 |
| 4 | A | 0.7061 | 0.5617 | 0.3850 | 0.5607 |
| 0 | P | 0.6677 | 0.5106 | 0.3768 | 0.5276 |
| 1 | P | 0.6485 | 0.5355 | 0.3708 | 0.5227 |
| 2 | P | 0.6547 | 0.5050 | 0.3604 | 0.5159 |
| 3 | P | 0.7172 | 0.5447 | 0.3887 | 0.5605 |
| 4 | P | 0.6974 | 0.5532 | 0.3797 | 0.5532 |


**PSNR(dB)**

| N | T | E | M | H | A |
| - | - | - | - | - | - |
| 0 | A | 23.4871 | 19.7800 | 20.5796 | 21.3970 |
| 1 | A | 23.4162 | 19.9195 | 20.6024 | 21.4229 |
| 2 | A | 22.6307 | 19.9171 | 20.1864 | 21.0036 |
| 3 | A | 24.0507 | 20.0122 | 20.8662 | 21.7686 |
| 4 | A | 24.0904 | 19.7361 | 20.9635 | 21.7240 |
| 0 | P | 23.5007 | 19.8589 | 20.5936 | 21.4318 |
| 1 | P | 23.8467 | 19.8510 | 20.8100 | 21.6238 |
| 2 | P | 23.5170 | 19.9682 | 20.5281 | 21.4530 |
| 3 | P | 24.0507 | 20.0122 | 20.8662 | 21.7686 |
| 4 | P | 24.2864 | 19.9264 | 20.9821 | 21.8640 |

**SSIM**

| N | T | E | M | H | A |
| - | - | - | - | - | - |
| 0 | A | 0.7854 | 0.5255 | 0.6207 | 0.6509 |
| 1 | A | 0.7772 | 0.5194 | 0.6078 | 0.6419 |
| 2 | A | 0.7581 | 0.5131 | 0.5893 | 0.6271 |
| 3 | A | 0.8049 | 0.5377 | 0.6334 | 0.6660 |
| 4 | A | 0.8190 | 0.5380 | 0.6401 | 0.6733 |
| 0 | P | 0.7817 | 0.5241 | 0.6172 | 0.6480 |
| 1 | P | 0.7916 | 0.5271 | 0.6207 | 0.6537 |
| 2 | P | 0.7748 | 0.5183 | 0.6060 | 0.6401 |
| 3 | P | 0.8049 | 0.5377 | 0.6334 | 0.6660 |
| 4 | P | 0.8161 | 0.5414 | 0.6397 | 0.6732 |

### Statistics

| model        | Params(M) | FLOPs(G) | FPS    | PSNR/Params(M) | ACC/Params(M) |
| ------------ | --------- | -------- | ------ | -------------- | ------------- |
| PSRec(srb=0) | 11.59     | 1.50     | 186.85 | 1.8492         | 0.0458        |
| PSRec(srb=1) | 11.71     | 1.63     | 153.13 | 1.8466         | 0.0465        |
| PSRec(srb=2) | 11.83     | 1.75     | 125.95 | 1.8134         | 0.0456        |
| PSRec(srb=3) | 11.95     | 1.88     | 108.87 | 1.8216         | 0.0469        |
| PSRec(srb=4) | 12.07     | 2.00     | 98.43  | 1.8114         | 0.0464        |

### Loss function

**Accuracy**

| loss | T | E      | M      | H      | A      |
| ---- | - | -      | -      | -      | -      |
| 0    | A | 0.6751 | 0.5319 | 0.3850 | 0.5397 |
| 0.5  | A | 0.6986 | 0.5553 | 0.3894 | 0.5573 |
| 1    | A | 0.7172 | 0.5447 | 0.3753 | 0.5605 |
| 5    | A | 0.6986 | 0.5553 | 0.3894 | 0.5573 |
| 0    | P | 0.0006 | 0.0    | 0.0    | 0.0002 |
| 0.5  | P | 0.6980 | 0.5496 | 0.3909 | 0.5557 |
| 1    | P | 0.7172 | 0.5447 | 0.3753 | 0.5605 |
| 5    | P | 0.6980 | 0.5496 | 0.3902 | 0.5555 |

**PSNR(dB)**

| loss | T | E      | M      | H      | A      |
| ---- | - | -      | -      | -      | -      |
| 0    | A | 19.6547 | 18.7143 | 18.7121 | 19.0615 |
| 0.5  | A | 23.4331 | 19.4744 | 20.8450 | 21.3600 |
| 1    | A | 24.0507 | 20.0122 | 20.8662 | 21.7686 |
| 5    | A | 23.4331 | 19.4744 | 20.8450 | 21.3600 |
| 0    | P | 21.4594 | 19.2615 | 19.5655 | 20.1680 |
| 0.5  | P | 23.8116 | 19.9792 | 20.8732 | 21.6716 |
| 1    | P | 24.0507 | 20.0122 | 20.8662 | 21.7686 |
| 5    | P | 23.8116 | 19.9792 | 20.8731 | 21.6716 |

**SSIM**

| loss | T | E      | M      | H      | A      |
| ---- | - | -      | -      | -      | -      |
| 0    | A | 0.5924 | 0.4247 | 0.4692 | 0.5004 |
| 0.5  | A | 0.8171 | 0.5396 | 0.6449 | 0.6746 |
| 1    | A | 0.8049 | 0.5377 | 0.6334 | 0.6660 |
| 5    | A | 0.8172 | 0.5396 | 0.6449 | 0.6746 |
| 0    | P | 0.7301 | 0.4960 | 0.5661 | 0.6041 |
| 0.5  | P | 0.8008 | 0.5332 | 0.6325 | 0.6627 |
| 1    | P | 0.8049 | 0.5377 | 0.6334 | 0.6660 |
| 5    | P | 0.8009 | 0.5332 | 0.6325 | 0.6627 |