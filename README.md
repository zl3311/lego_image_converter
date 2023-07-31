# lego_image_converter
 A simple program that converts an image to a Lego-fied picture.

---

## Background
Here's how this simple package started. 
(Feel free to skip if you're in a hurry, but why would anyone trying to enjoy Lego and to kill time in a such hurry?)

Originally, I purchased [Lego Art Project 21226](https://www.lego.com/en-us/product/art-project-create-together-21226)
(now retired) as the Thanksgiving gift for myself back in 2022, but I didn't have the chance to 
put up the pieces, since recession is kind of suffocating many folks, including myself.

It is until mid 2023 that I finally have some room to breath, and I realize that I haven't opened this toy that 
has been sitting in the corner of my room 
for like 8 months or so. 
So I reckon it's probably a decent timing to bring some innocent joy back to my life. 

The main bummer is that Lego includes barely enough blocks only to build the 
images in their manuals 
(like blue blocks are the most common (660 pieces), but considering a 48x48=2304 canvas, it means you need colorful 
background otherwise you'll run out of some color blocks), and that is not enough if you want to do something fancy. 
I understand that Lego is for-profit company, and real hardcore Lego players spend tons of money 
on buying customized blocks themselves (and I'm not prodigal enough to be one of them). 
Still, I feel like Lego is not generous enough for such a brilliant idea on this Art Project product. 

Another issue is that there isn't a tool that automatically converts a random image into a prototype as if you were to 
build it in Lego, which means you might spend tons of time working on something that is undesirable, 
and we adults are short on time. This is the reason why I build this package. You're welcome, Lego Engineering team.

Don't get me wrong: I'm still a huge fan of Lego, 
and I hope this package can save Lego players and likely programmers as well some time prototyping their ideas.

## How to use
Please install the package using 

`pip install lego_image_converter`

You can check out some examples on [this Colab notebook](https://colab.research.google.com/drive/17k9ckWLznP_u6kH2rp3Ibujk6voFmn-X#scrollTo=nS3hGf1_kpy4). 

**Disclaimer: I do not own any right of the pictures in examples, 
and they are not intended for profit either. If you're the owner of the pictures
and would like me to remove it, feel free to reach out, and I'm happy to remove it for you.** 

I screenshot the pictures from the Internet, namely
- Tsunami from [here](https://www.zazzle.com/the_great_wave_off_kanagawa_8_bit_pixel_art_plaque-200560049736055877).
- Cheetah's logo from [here](https://is3-ssl.mzstatic.com/image/thumb/Purple125/v4/73/dd/09/73dd0955-fabe-9a96-cc09-641dbf9b9141/source/512x512bb.jpg).
- Twitter's logo from [here](https://commons.wikimedia.org/wiki/File:Logo_of_Twitter.svg).
- Doge's logo from [here](https://variety.com/2023/digital/news/elon-musk-twitter-logo-doge-dogecoin-meme-1235572343/).

## Notes

- The trimming procedure of this package is super dumb, because it's too tedious to do the CV stuff
  (like figure out the center of the picture, majorly and minorly adjust the anchoring point, etc.) 
 for such a mini package and I'm not a CV expert. 
My suggestion for you is to start your screenshot from the lower left corner, drag it until 
either the length or width dimension is desirable and leave extra buffer in the other dimension. 
The trimming algorithm can only handle this situation properly, so it is not intelligent at all.

- I was using a naive Euclidean distance on RGB channel for matching pixels and Lego colors, and it sucks. 
I found this [colormath](https://python-colormath.readthedocs.io/en/latest/) package useful in 
addressing the nonlinearities of human vision perception, and it performs decently well.

- I'm using `unlimited_blocks=True` for the sake of illustration.