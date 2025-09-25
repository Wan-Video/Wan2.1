# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models), Hunyuan Video and LTV Video models with:
- Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
- Support for old GPUs (RTX 10XX, 20xx, ...)
- Very Fast on the latest GPUs
- Easy to use Full Web based interface
- Auto download of the required model adapted to your specific architecture
- Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation, MMAudio, Video Browser, Pose / Depth / Flow extractor
- Loras Support to customize each model
- Queuing system : make your shopping list of videos to generate and come back later 

**Discord Server to get Help from Other Users and show your Best Videos:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

## 🔥 Latest Updates : 
### September 25 2025: WanGP v8.73 - Here Are ~~Two~~Three New Contenders in the Vace Arena !

So in today's release you will find two Wannabe Vace that covers each only a subset of Vace features but offers some interesting advantages:
- **Wan 2.2 Animate**: this model is specialized in *Body Motion* and *Facial Motion transfers*. It does that very well. You can use this model to either *Replace* a person in an in Video or *Animate* the person of your choice using an existing *Pose Video* (remember *Animate Anyone* ?). By default it will keep the original soundtrack. *Wan 2.2 Animate* seems to be under the hood a derived i2v model and should support the corresponding Loras Accelerators (for instance *FusioniX t2v*). Also as a WanGP exclusivity, you will find support for *Outpainting*.

In order to use Wan 2.2 Animate you will need first to stop by the *Mat Anyone* embedded tool, to extract the *Video Mask* of the person from which you want to extract the motion.

- **Lucy Edit**: this one claims to be a *Nano Banana* for Videos. Give it a video and asks it to change it (it is specialized in clothes changing) and voila ! The nice thing about it is that is it based on the *Wan 2.2 5B* model and therefore is very fast especially if you the *FastWan* finetune that is also part of the package.

Also because I wanted to spoil you:
- **Qwen Edit Plus**: also known as the *Qwen Edit 25th September Update* which is specialized in combining multiple Objects / People. There is also a new support for *Pose transfer* & *Recolorisation*. All of this made easy to use in WanGP. You will find right now only the quantized version since HF crashes when uploading the unquantized version.

*Update 8.71*: fixed Fast Lucy Edit that didnt contain the lora
*Update 8.72*: shadow drop of Qwen Edit Plus
*Update 8.73*: Qwen Preview & InfiniteTalk Start image 

### September 15 2025: WanGP v8.6 - Attack of the Clones

- The long awaited **Vace for Wan 2.2** is at last here or maybe not: it has been released by the *Fun Team* of *Alibaba* and it is not official. You can play with the vanilla version (**Vace Fun**) or with the one accelerated with Loras (**Vace Fan Cocktail**)

- **First Frame / Last Frame for Vace** : Vace models are so powerful that they could do *First frame / Last frame* since day one using the *Injected Frames* feature. However this required to compute by hand the locations of each end frame since this feature expects frames positions. I made it easier to compute these locations by using the "L" alias :

For a video Gen from scratch *"1 L L L"* means the 4 Injected Frames will be injected like this: frame no 1 at the first position, the next frame at the end of the first window, then the following frame at the end of the next window, and so on ....
If you *Continue a Video* , you just need *"L L L"* since the first frame is the last frame of the *Source Video*. In any case remember that numeral frames positions (like "1") are aligned by default to the beginning of the source window, so low values such as 1 will be considered in the past unless you change this behaviour in *Sliding Window Tab/ Control Video, Injected Frames aligment*.

- **Qwen Edit Inpainting** exists now in two versions: the original version of the previous release and a Lora based version. Each version has its pros and cons. For instance the Lora version supports also **Outpainting** ! However it tends to change slightly the original image even outside the outpainted area.

- **Better Lipsync with all the Audio to Video models**: you probably noticed that *Multitalk*, *InfiniteTalk* or *Hunyuan Avatar* had so so lipsync when the audio provided contained some background music. The problem should be solved now thanks to an automated background music removal all done by IA. Don't worry you will still hear the music as it is added back in the generated Video.

### September 11 2025: WanGP v8.5/8.55 - Wanna be a Cropper or a Painter ?

I have done some intensive internal refactoring of the generation pipeline to ease support of existing models or add new models. Nothing really visible but this makes WanGP is little more future proof.

Otherwise in the news:
- **Cropped Input Image Prompts**: as quite often most *Image Prompts* provided (*Start Image, Input Video, Reference Image,  Control Video, ...*) rarely matched your requested *Output Resolution*. In that case I used the resolution you gave either as a *Pixels Budget* or as an *Outer Canvas* for the Generated Video. However in some occasion you really want the requested Output Resolution and nothing else. Besides some models deliver much better Generations if you stick to one of their supported resolutions. In order to address this need I have added a new Output Resolution choice in the *Configuration Tab*:  **Dimensions Correspond to the Ouput Weight & Height as the Prompt Images will be Cropped to fit Exactly these dimensins**. In short if needed the *Input Prompt Images* will be cropped (centered cropped for the moment). You will see this can make quite a difference for some models

- *Qwen Edit* has now a new sub Tab called **Inpainting**, that lets you target with a brush which part of the *Image Prompt* you want to modify. This is quite convenient if you find that Qwen Edit modifies usually too many things. Of course, as there are more constraints for Qwen Edit don't be surprised if sometime it will return the original image unchanged. A piece of advise: describe in your *Text Prompt* where (for instance *left to the man*, *top*, ...) the parts that you want to modify are located.

The mask inpainting is fully compatible with *Matanyone Mask generator*: generate first an *Image Mask* with Matanyone, transfer it to the current Image Generator and modify the mask with the *Paint Brush*. Talking about matanyone I have fixed a bug that caused a mask degradation with long videos (now WanGP Matanyone is as good as the original app and still requires 3 times less VRAM)

- This **Inpainting Mask Editor** has been added also to *Vace Image Mode*. Vace is probably still one of best Image Editor today. Here is a very simple & efficient workflow that do marvels with Vace:
Select *Vace Cocktail > Control Image Process = Perform Inpainting & Area Processed = Masked Area > Upload a Control Image, then draw your mask directly on top of the image & enter a text Prompt that describes the expected change > Generate > Below the Video Gallery click 'To Control Image' > Keep on doing more changes*.

Doing more sophisticated thing Vace Image Editor works very well too: try Image Outpainting, Pose transfer, ...

For the best quality I recommend to set in *Quality Tab* the option: "*Generate a 9 Frames Long video...*" 

**update 8.55**: Flux Festival
- **Inpainting Mode** also added for *Flux Kontext*
- **Flux SRPO** : new finetune with x3 better quality vs Flux Dev according to its authors. I have also created a *Flux SRPO USO* finetune which is certainly the best open source *Style Transfer* tool available
- **Flux UMO**: model specialized in combining multiple reference objects / people together. Works quite well at 768x768

Good luck with finding your way through all the Flux models names !

### September 5 2025: WanGP v8.4 - Take me to Outer Space
You have probably seen these short AI generated movies created using *Nano Banana* and the *First Frame - Last Frame* feature of *Kling 2.0*. The idea is to generate an image, modify a part of it with Nano Banana and give the these two images to Kling that will generate the Video between these two images, use now the previous Last Frame as the new First Frame, rinse and repeat and you get a full movie.

I have made it easier to do just that with *Qwen Edit* and *Wan*:
- **End Frames can now be combined with Continue a Video** (and not just a Start Frame)
- **Multiple End Frames can be inputed**, each End Frame will be used for a different Sliding Window

You can plan in advance all your shots (one shot = one Sliding Window) : I recommend using Wan 2.2 Image to Image with multiple End Frames (one for each shot / Sliding Window), and a different Text Prompt for each shot / Sliding Winow (remember to enable *Sliding Windows/Text Prompts Will be used for a new Sliding Window of the same Video Generation*)

The results can quite be impressive. However, Wan 2.1 & 2.2 Image 2 Image are restricted to a single overlap frame when using Slide Windows, which means only one frame is reeused for the motion. This may be unsufficient if you are trying to connect two shots with fast movement.

This is where *InfinitTalk* comes into play. Beside being one best models to generate animated audio driven avatars, InfiniteTalk uses internally more one than motion frames. It is quite good to maintain the motions between two shots. I have tweaked InfinitTalk so that **its motion engine can be used even if no audio is provided**.
So here is how to use InfiniteTalk: enable *Sliding Windows/Text Prompts Will be used for a new Sliding Window of the same Video Generation*), and if you continue an existing Video  *Misc/Override Frames per Second" should be set to "Source Video*. Each Reference Frame inputed will play the same role as the End Frame except it wont be exactly an End Frame (it will correspond more to a middle frame, the actual End Frame will differ but will be close)


You will find below a 33s movie I have created using these two methods. Quality could be much better as I havent tuned at all the settings (I couldn't bother, I used 10 steps generation without Loras Accelerators for most of the gens).

### September 2 2025: WanGP v8.31 - At last the pain stops

- This single new feature should give you the strength to face all the potential bugs of this new release:
**Images Management (multiple additions or deletions, reordering) for Start Images / End Images / Images References.**  

- Unofficial **Video to Video (Non Sparse this time) for InfinitTalk**. Use the Strength Noise slider to decide how much motion of the original window you want to keep. I have also *greatly reduced the VRAM requirements for Multitalk / Infinitalk* (especially the multispeakers version & when generating at 1080p). 

- **Experimental Sage 3 Attention support**: you will need to deserve this one, first you need a Blackwell GPU (RTX50xx) and request an access to Sage 3 Github repo, then you will have to compile Sage 3, install it and cross your fingers ...


*update 8.31: one shouldnt talk about bugs if one doesn't want to attract bugs*


See full changelog: **[Changelog](docs/CHANGELOG.md)**

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)

## 🚀 Quick Start

**One-click installation:** Get started instantly with [Pinokio App](https://pinokio.computer/)

**Manual installation:**
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

**Run the application:**
```bash
python wgp.py  # Text-to-video (default)
python wgp.py --i2v  # Image-to-video
```

**Update the application:**
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
```bash
git pull
pip install -r requirements.txt
```


## 📦 Installation

For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for RTX 10XX to RTX 50XX

## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities

### Advanced Features
- **[Loras Guide](docs/LORAS.md)** - Using and managing Loras for customization
- **[Finetunes](docs/FINETUNES.md)** - Add manually new models to WanGP
- **[VACE ControlNet](docs/VACE.md)** - Advanced video control and manipulation
- **[Command Line Reference](docs/CLI.md)** - All available command line options

## 📚 Documentation

- **[Changelog](docs/CHANGELOG.md)** - Latest updates and version history
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 📚 Video Guides
- Nice Video that explain how to use Vace:\
https://www.youtube.com/watch?v=FMo9oN2EAvE
- Another Vace guide:\
https://www.youtube.com/watch?v=T5jNiEhf9xk

## 🔗 Related Projects

### Other Models for the GPU Poor
- **[HuanyuanVideoGP](https://github.com/deepbeepmeep/HunyuanVideoGP)** - One of the best open source Text to Video generators
- **[Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)** - Image to 3D and text to 3D tool
- **[FluxFillGP](https://github.com/deepbeepmeep/FluxFillGP)** - Inpainting/outpainting tools based on Flux
- **[Cosmos1GP](https://github.com/deepbeepmeep/Cosmos1GP)** - Text to world generator and image/video to world
- **[OminiControlGP](https://github.com/deepbeepmeep/OminiControlGP)** - Flux-derived application for object transfer
- **[YuE GP](https://github.com/deepbeepmeep/YuEGP)** - Song generator with instruments and singer's voice

---

<p align="center">
Made with ❤️ by DeepBeepMeep
</p> 
