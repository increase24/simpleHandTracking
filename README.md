# handGesRecDLL
Hybrid C++/C# visual studio solution for isolated hand gesture recognition.
The C++ project is for dll generation and the C# project is for dll function calling.

The **onnxruntime** is employed as the inference engine, otherwise, we adopt **opencv** for image matrix processing and **libtorch** for tensor computing.  

## dependency
+ c10.dll
+ **torch.dll**
+ **onnxruntime.dll**
+ openblas.dll
+ opencv_world410.dll
+ libiomp5md.dll
+ libiompstubs5md.dll
+ libomp.dll
+ libsodium.dll
+ libzmq.dll
+ flang.dll
+ flangrti.dll
