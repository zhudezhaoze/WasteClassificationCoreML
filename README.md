# Waste Classification App Using CoreML
A simple yet useful IOS application that can visually recognizes waste types and give instructions on garbage disposal. 

**1. Why I Do This?**

Waste collection and rubbish disposal play an extremely important role in the global cleanliness and sustainability drive, with people’s health and the conservation of resources being the responsibility of everyone. The application uses computer vision and machine learning technology to help people to classify wastes quickly. With this mobile application, people can simply scan the waste to be disposed, and follow the instruction provided by the app to dispose the waste to the right type of garbage disposal. This is an useful and handy application especially for people who live in the countries that have evolving yet strict waste classificatin policies to comply. 
![b7d41aa04e4da12ee42f355b9c8e1b10_XL](https://user-images.githubusercontent.com/60851886/95681222-4380f900-0ba4-11eb-91e9-f22d9cf02f5f.jpg)

**2. Model Training**

Azure Custom Vision and Core ML were the two major tools I used to train and deploy the computer vision model in the application. In this app, I used Custom Vision to train a classification model and converted it into a Core ML model before the deployment in the following steps:

  1. Collect images! I collected 100 images for each of 20 different wastes in various lightings, various positions on the phone screen, with different backgrounds. Here are the snapshots of the image collections! The training dataset can be found from here: https://www.kaggle.com/wangziang/waste-pictures
 ![Screen Shot 2020-10-11 at 10 02 17 AM](https://user-images.githubusercontent.com/60851886/95682090-0a975300-0ba9-11eb-8f53-3dcd1f36a842.png)

  2. Upload images and add tags. After image sets have been added and tagged, click the Train button at the top to let Custom Vision’s machine learning engine trains a model using the images being fed.
  
  3. Export the trained model as a Core ML model by hitting the Export button and then selecting the export type to be iOS — Core ML.
 
 **3. Model Deployment & Codes Implementation**
 
The steps below show the steps I took to connect the camera:

1. To enable the computer to recognize the target, we need to keep the image updating. For this reason, the live images must be fed to the Core ML model for prediction.

```
func configureCamera() {
 
     //Start capture session
     let captureSession = AVCaptureSession()
     captureSession.sessionPreset = .photo
     captureSession.startRunning()
 
     // Add input for capture
     guard let captureDevice = AVCaptureDevice.default(for: .video) else { return }
     guard let captureInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
     captureSession.addInput(captureInput)
 
     // Add preview layer to our view to display the open camera screen
     let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
     view.layer.addSublayer(previewLayer)
     previewLayer.frame = view.frame
 
     // Add output of capture
     /* Here we set the sample buffer delegate to our viewcontroller whose callback 
         will be on a queue named - videoQueue */
     let dataOutput = AVCaptureVideoDataOutput()
     dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
     captureSession.addOutput(dataOutput)
 }
 
 ```
 
2. Set the video output buffers. 

```
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
...
...
...
}
```

3. One last thing is to set up the simulator so that while deploying to a real device, we can get the permission to use the camera. This can be done in the info.plist page.

For model deployment, I followed the next few steps:

1. Import the Vision library and import the Core ML.
```
import CoreML
import Vision
```

2. Initiate an enum object to store all the labels used in the model.

```
enum wastes: String {
case bandaid = "Bandaid"
case battery = "Battery"
case bread = "Bread"
case bulb = "Bulb"
case cans = "Cans"
case carton = "Carton"
case chopsticks = "Chopsticks"
case diapers = "Diapers"
case dishes = "Dishes"
case facialmask = "FacialMask"
case glassbottle = "GlassBottle"
case leftovers = "Leftovers"
case medicinebottle = "MedicineBottle"
case milkbox = "Milkbox"
case napkin = "Napkin"
case newspaper = "Newspaper"
case nuts = "Nuts"
case pen = "Pen"
case plasticbag = "PlasticBag"
case plasticbottle = "PlassticBottle"
}
```

4. I then add a UILabel named “predictionLabel” and set a textbox in the bottom of the screen to show the prediction results.

```
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
 
 @IBOutlet weak var predictionLabel: UILabel!
 
 override func viewDidLoad() {
     super.viewDidLoad()
     configureCamera()
 }
 
 override func didReceiveMemoryWarning() {
     super.didReceiveMemoryWarning()
 }
 ```
 
 5. After all the preparation are done, it is time to initialize the Core ML model –wastesmodel.mlmodel- and get the buffer input from AVCaptureConnection and feed it to the wastes detector model to get the output of its prediction by using the method we build.
 
 ```
 func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
 
     // Initialise CVPixelBuffer from sample buffer
     guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
 
     //Initialise Core ML model
     guard let wasteModel = try? VNCoreMLModel(for: wastemodel().model) else { return }
 
     // Create a Core ML Vision request
     let request =  VNCoreMLRequest(model: wasteModel) { (finishedRequest, err) in
 
         // Dealing with the result of the Core ML Vision request
         guard let results = finishedRequest.results as? [VNClassificationObservation] else { return }
 
         guard let firstResult = results.first else { return }
         var predictionString = ""
         DispatchQueue.main.async {
             switch firstResult.identifier {
             case "bandaid":
                 predictionString = "(Residual Waste)"
             case "battery":
                 predictionString = "(Hazardous Waste)"
             case "bread":
                 predictionString = "(Food Waste)"
             case "bulb":
                 predictionString = "(Hazardous Waste)"
             case "cans":
                 predictionString = "(Recyclable)"
             case "carton":
                 predictionString = "(Recyclable)"
             case "chopsticks":
                 predictionString = "(Residual Waste)"
             case "diapers":
                 predictionString = "(Residual Waste)"
             case "dishes":
                 predictionString = "(Residual Waste)"
             case "facialmask":
                 predictionString = "(Residual Waste)"
             case "glassbottle":
                 predictionString = "(Recyclable)"
             case "leftovers":
                 predictionString = "(Food Waste)"
             case "medicinebottle":
                 predictionString = "(Recyclable)"
             case "milkbox":
                 predictionString = "(Recyclable)"
             case "napkin":
                 predictionString = "(Residual Waste)"
             case "newspaper":
                 predictionString = "(Recyclable)"
             case "nuts":
                 predictionString = "(Recyclable)"
             case "pen":
                 predictionString = "(Residual Waste)"
             case "plasticbag":
                 predictionString = "(Recyclable)"
             case "plasticbottle":
                 predictionString = "(Recyclable)"
             default:
                 predictionString = "(Residual Waste)"
             }
 
             self.predictionLabel.text = firstResult.identifier + predictionString + "(\(firstResult.confidence))" 
         }
     }
 
     // Perform the above request using Vision Image Request Handler
     try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
 }
```

**4. Test Results & Demonstration**

Based on a wild range of item experiments, we conclude that the model predictions have a high accuracy on recognizing type of wastes that are included in the category. Particularly, it can accurately differentiate chopsticks from pens, even though they look similar in shape. However, it sometimes mistakes lamp with facial masks, which is likely due to the lack of diversity in training data. 

![giphy](https://user-images.githubusercontent.com/60851886/96681683-4cf52880-133d-11eb-903e-30e5c5d5efee.gif)


**5. Further Improvement**

Although we found that the model can accurately identify all items that are in the designated categories, the confidence of its predictions dropped significantly when the camera moves away from the obejct. It may be caused due to the training data not including object images captured from distance. Besides, the model can only recognize 20 items so far, which may show its limitations while being used to recognize and classify a variety of waste items. Thus, in the future, we decide to improve the prediction accuracy by using images that are taken from wild ranges. In addition, we will keep adding more types of waste items to the training data to allow the model to recognize a more diverse set of wastes.


