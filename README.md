# UNet-Transformer

 UNEt TRansformers (UNETR),
 utilizes a transformer as the encoder to learn sequence
 representations of the input volume and effectively capture
 the global multi-scale information, while also following the
 successful “U-shaped” network design for the encoder and
 decoder. The transformer encoder is directly connected to
 a decoder via skip connections at different resolutions to
 compute the final semantic segmentation output. 

  The task of 3D segmentation is formulated as a 1D sequence-to-sequence prediction problem
 and a vision transformer is used as the encoder to learn contextual
 information from the embedded input patches. The extracted
 representations from the transformer encoder are merged
 with the CNN-baseddecoder via skip connections at multiple
 resolutions to predict the segmentation outputs. Instead of
 using transformers in the decoder, the framework
 uses a CNN-based decoder. This is due to the fact that transformers are unable to properly capture localized information,
 despite their great capability of learning global information.

 # Dataset Used 
 The Multi Atlas Labeling Beyond The Cranial Vault (BTCV)dataset consists of 30subjects with abdominal CTscans
 where 13 organs were annotated by interpreters under supervision 
 of clinical radiologists at Vanderbilt University MedicalCenter.
 Each CT scan was acquired with contrast enhancement in portal venous 
 phase and consists of 80 to 225 slices with 512x512 pixels and slice
 thickness ranging from 1 to 6mm. Each volume has been
 pre-processed independently by normalizing the intensities
 in the range of[-1000,1000] HU to [0,1]. All image sare
 resampled into the isotropic voxel spacingof 1.0mm during
 pre-processing.

 # Evalulation
 The loss function used is a combination of soft dice loss
 and cross-entropy loss.

 # Training Process
 The model was trained with the batch size of 2, using the AdamW 
 optimizer with initial learning rate of 0.0001 for 20 iterations.

 # Original paper
 https://arxiv.org/abs/2103.10504

 # Original Implementation
 https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV
