//
//  ViewController.swift
//  CustomVisionMicrosoftToCoreML
//
//  Created by Sayalee on 6/28/18.
//  Copyright © 2018 Assignment. All rights reserved.
//

import UIKit
import AVKit
import Vision

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

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    @IBOutlet weak var predictionLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        configureCamera()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func configureCamera() {
        
        //Start capture session
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        captureSession.startRunning()
        
        // Add input for capture
        guard let captureDevice = AVCaptureDevice.default(for: .video) else { return }
        guard let captureInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        captureSession.addInput(captureInput)
        
        // Add preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        // Add output for capture
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(dataOutput)
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
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

}

