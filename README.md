# HMMs-for-Barecode-decoding
Use Hidden Markov Models (HMMs) to decode barcodes

The aim of this project is to develop a probabilistic model for modeling a specific type linear barcodes, called
'UPC-A'. Our aim will be to infer the subsequent digits (we will sometimes refer to them as symbols) given a barcode
scanline that is obtained from a gray-scale image. This is a challenging task since the observed scanline can be degraded
due to noise and blur which often occur in practice.
