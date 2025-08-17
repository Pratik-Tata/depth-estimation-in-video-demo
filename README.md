# depth-estimation-in-video-demo
A small demo project that uses MiDas tiny ONNX model to convert a 2D video into 3D(ish)

The project needs a Depth estimation model for it. The model can be downloaded from hugging face, for this particular demp, use a tiny model like MiDas_Tiny. Since this project runs on browser and uses HTML as interface, the model must be tiny as bigger models may cause an error because browsers may not support int64. The model must be in onnx format. 
