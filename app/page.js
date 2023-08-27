'use client'
import LABELS from '../datasets/classes.json'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import { useEffect, useRef, useState } from 'react'
import './page.module.css'


const ZOO_MODEL = [{ name: 'yolov5', child: ['yolov5n', 'yolov5s'] }]

function RootPage() {
    const [model, setModel] = useState(null);
    const [aniId, setAniId] = useState(null);
    const [modelName, setModelName] = useState(ZOO_MODEL[0]);
    const [loading, setLoading] = useState(0)
    const [isCatDetected, setIsCatDetected] = useState(false)

    const videoRef = useRef(null)
    const canvasRef = useRef(null)

    const [singleImage, setSingleImage] = useState(false)
    const [liveWebcam, setLiveWebcam] = useState(false)

    useEffect(() => {
        tf.loadGraphModel(`model/${modelName.name}/${modelName.child[1]}/model.json`, {
            onProgress: (fractions) => {
                setLoading(fractions)
            }
        }).then(async (mod) => {
            const dummy = tf.ones(mod.inputs[0].shape)
            const res = await mod.executeAsync(dummy)
            tf.dispose(res)
            tf.dispose(dummy)
            setModel(mod)
        })
    }, [modelName])

    const renderPrediction = (boxesData, scoresData, classesData) => {
        const ctx = canvasRef.current.getContext('2d')
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        const font = '16px sans-serif'
        ctx.font = font
        ctx.textBaseline = 'top'

        for (let i = 0; i < scoresData.length; ++i) {
            const klass = LABELS[classesData[i]]
            const score = (scoresData[i] * 100).toFixed(1)
            let [x1, y1, x2, y2] = boxesData.slice(i * 4, (i + 1) * 4)
            x1 *= canvasRef.current.width
            x2 *= canvasRef.current.width
            y1 *= canvasRef.current.height
            y2 *= canvasRef.current.height
            const width = x2 - x1
            const height = y2 - y1
            ctx.strokeStyle = '#003B5F'
            ctx.lineWidth = 2
            ctx.strokeRect(x1, y1, width, height)
            const label = klass + ' - ' + score + '%'
            const textWidth = ctx.measureText(label).width
            const textHeight = parseInt(font, 10)
            ctx.fillStyle = '#003B5F'
            ctx.fillRect(x1 - 1, y1 - (textHeight + 4), textWidth + 6, textHeight + 4)
            ctx.fillStyle = '#FFFFFF'
            ctx.fillText(label, x1 + 2, y1 - (textHeight + 2))

            if (klass === 'cat') {
                setIsCatDetected(true)
            }
        }
    }

    const doPredictFrame = async () => {
        if (!model) return
        if (!videoRef.current || !videoRef.current.srcObject) return
        tf.engine().startScope()
        const [modelWidth, modelHeight] = model.inputs[0].shape.slice(1, 3)
        const input = tf.tidy(() => {
            const frameTensor = tf.browser.fromPixels(videoRef.current)
            return tf.image.resizeBilinear(frameTensor, [modelWidth, modelHeight]).div(255.0).expandDims(0)
        })

        const res = await model.executeAsync(input)
        const [boxes, scores, classes] = res
        const boxesData = boxes.dataSync()
        const scoresData = scores.dataSync()
        const classesData = classes.dataSync()
        renderPrediction(boxesData, scoresData, classesData)
        tf.dispose(res)
        const reqId = requestAnimationFrame(doPredictFrame)
        setAniId(reqId)
        tf.engine().endScope()
    }

    const webcamHandler = async () => {
        if (liveWebcam) {
            cancelAnimationFrame(aniId)
            videoRef.current.srcObject.getTracks().forEach(track => track.stop())
            videoRef.current.srcObject = null
        } else {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return

            const media = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: {
                    facingMode: 'environment'
                }
            })

            window.localStream = media
            videoRef.current.srcObject = media
            videoRef.current.onloadedmetadata = () => {
                doPredictFrame()
            }
            setLiveWebcam(prev => !prev)
        }
    }

    useEffect(() => {
        if (isCatDetected) {
            const timerId = setTimeout(() => {
                setIsCatDetected(false)
            }, 6000)

            return () => {
                clearTimeout(timerId)
            }
        }
    }, [isCatDetected])

    return (
        <>
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <h1>
                    Tensorflow.js Example
                </h1>
                <video
                    ref={videoRef}
                    className={`video ${liveWebcam && !singleImage ? 'block' : 'hidden'}`}
                    autoPlay
                    playsInline
                    muted
                    style={{ position: 'absolute', borderRadius: '15px' }}
                />
                <canvas
                    style={{
                        position: 'relative',
                        zIndex: 99999999,
                        top: '-20px'
                    }}
                    ref={canvasRef}
                    width={540}
                    height={500}
                />
                {isCatDetected && (
                    <div style={{ marginTop: '10px', color: 'red', fontWeight: 'bold', fontFamily: 'Nunito', marginBottom: '10px' }}>
                        <span>Hay un gatito</span>
                    </div>
                )}
                <div>
                    {model ?
                        <button className='primay-button' onClick={webcamHandler}>
                            {liveWebcam ? 'Stop Webcam' : 'Start Webcam'}
                        </button> : 'Modelo Cargando...'
                    }

                </div>
            </div>
        </>
    )
}

export default RootPage
