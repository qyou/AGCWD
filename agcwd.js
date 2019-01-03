/** 
 * adaptive gamma correction based on the reference.
 * Reference:
 *   S. Huang, F. Cheng and Y. Chiu, "Efficient Contrast Enhancement Using Adaptive Gamma Correction With
 *   Weighting Distribution," in IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041,
 *   March 2013. doi: 10.1109/TIP.2012.2226047
 * Revised from
 *      https://github.com/mss3331/AGCWD/blob/master/AGCWD.m
 */
const cv = require('opencv4nodejs')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

function _extractValueChannel(normalizedColorImage) {
    const hsv = normalizedColorImage.cvtColor(cv.COLOR_BGR2HSV)
    const hsvChannels = hsv.split()
    const valueChannel = hsvChannels[2]
    return valueChannel.mul(255).convertTo(cv.CV_8U)
}

function _setValueChannel(colorImage, valueChannel) {
    const normalizedValueChannel = valueChannel.convertTo(cv.CV_32F).div(255)
    const normalizedColorImage = colorImage.convertTo(cv.CV_32F).div(255)
    const hsv = normalizedColorImage.cvtColor(cv.COLOR_BGR2HSV)
    const [hue, saturation] = hsv.split().slice(0, 2)
    return new cv.Mat([hue, saturation, normalizedValueChannel]).cvtColor(cv.COLOR_HSV2BGR).mul(255).convertTo(cv.CV_8U)
}

function _getPDF(image) {
    const {
        cols,
        rows
    } = image
    const pixelCount = cols * rows
    const hist = cv.calcHist(image, [{
        channel: 0,
        bins: 256,
        ranges: [0, 255]
    }])
    return hist.convertTo(cv.CV_32F).div(pixelCount)
}

function _enhance(image, pdf) {
    const pdfTensor = tf.tensor(pdf.getDataAsArray())
    const cdfTensor = tf.cumsum(pdfTensor).div(tf.sum(pdfTensor))
    const cdfArray = cdfTensor.squeeze().dataSync()
    const intensityArray = cdfArray.map((value, index) => 255 * (index / 255) ** (1 - value))
    const {
        cols,
        rows
    } = image
    console.log(image)
    const enhancedImage = image.copy()
    console.log(enhancedImage)
    for (let i = 0; i < rows; ++i) {
        for (let j = 0; j < cols; ++j) {
            const intensity = enhancedImage.atRaw(i, j)
            enhancedImage.set(i, j, [intensityArray[intensity]])
        }
    }
    return enhancedImage
}

function _transform(image) {
    const imagePDF = _getPDF(image)
    const {
        minVal,
        maxVal
    } = imagePDF.minMaxLoc()
    const {
        cols,
        rows
    } = imagePDF
    const minMat = new cv.Mat(rows, cols, cv.CV_32FC1, minVal)
    // use `sqrt` as pow(element, weight=0.5)
    // you can try to revised the next line 
    const weightedImagePDF = (imagePDF.sub(minMat).div(maxVal - minVal)).sqrt().mul(maxVal)
    return _enhance(image, weightedImagePDF)
}

function transformByAGCWD(image) {
    let img = image.copy()
    const isColor = img.channels >= 3
    if (isColor) {
        const normalizedColorImage = img.convertTo(cv.CV_32F).div(255)
        img = _extractValueChannel(normalizedColorImage)
    }
    let enhancedImage = _transform(img)
    if (isColor) {
        enhancedImage = _setValueChannel(image, enhancedImage)
    }
    return enhancedImage
}

module.exports = {
    transformByAGCWD
}