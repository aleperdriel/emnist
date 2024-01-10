window.addEventListener('load', () => {
    canvas = document.getElementById('draw_box');    
    ctx = canvas.getContext('2d');    
    ctx.fillStyle = "black";    
    ctx.fillRect(0, 0, canvas.width, canvas.height); 


    document.addEventListener('mousedown', startDrawing); 
    document.addEventListener('mouseup', stopDrawing); 
    document.addEventListener('mousemove', sketch); 

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = new Float32Array(imageData.data.length / 4);
    isDrawing = false


    document.getElementById('clear').addEventListener("click", () => {  
    ctx.clearRect(0, 0, canvas.width, canvas.height);  
    ctx.fillStyle = "black"; 
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});
})
let coord = {x:0 , y:0};  


  const startDrawing = (event) => {
    isDrawing = true;
    getPosition(event); 
  }

  const stopDrawing = () => {
    isDrawing = false
    runModel()
  }

  function getPosition(event){ 
    coord.x = event.clientX - canvas.offsetLeft; 
    coord.y = event.clientY - canvas.offsetTop; 
  } 

  const sketch = (event) => {
    if(!isDrawing) return;

    ctx.beginPath(); 

    ctx.strokeStyle = 'white'; 
    ctx.lineWidth = 10; 
    ctx.lineCap = 'round'; 

    ctx.moveTo(coord.x, coord.y); 
    getPosition(event); 
    ctx.lineTo(coord.x , coord.y); 

    ctx.stroke(); 
  }
  

  
  const updateImgData = () => {
    const newData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0, j = 0; i < newData.length; i += 4, ++j) {
        data[j] = (255 - newImageData[i]) / 255.0;
    }
  }

  const softmax = (data) => {
    const exps = data.map((value) => Math.exp(value))
    const sumExps = exps.reduce((acc, val) => acc + val)
    return exps.map((exp) => exp / sumExps)
  }

  const runModel = async () => {
    const session = await ort.InferenceSession.create('./emnist.onnx')
    const data = Float32Array.from(data).map((pixel) => (pixel - 0.1307) / 0.3081)
    const input = new ort.Tensor('float32', data, [1, 1, 28, 28])
    const result = await session.run({ 'input': input })
    const logits = result.output.data
    const probas = softmax(logits)
    
    let maxKey, maxValue = 0;

    for(const [key, value] of Object.entries(probas)) {
      if(value > maxValue) {
        maxValue = value;
        maxKey = key;
      }
    }

    // console.log("=== Result", probas)
    // topResult = document.getElementById('result').innerText = (maxValue + 9).toString(36).toUpperCase()

    console.log("=== Result", topResult)
    
  }

  
  runModel()