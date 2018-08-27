/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.util.codec;

import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelInputs;
import com.amazonaws.ml.mms.util.messages.ModelLoadModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageCodec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;


@ChannelHandler.Sharable
public class MessageEncoder extends ByteToMessageCodec<BaseModelRequest> {
    private static final Logger logger = LoggerFactory.getLogger(MessageEncoder.class);

    private void encodeRequestBatch(RequestBatch req, ByteBuf out) {
        out.writeInt(req.getRequestId().length());
        out.writeBytes(req.getRequestId().getBytes());

        if(req.getContentType() != null) {
            out.writeInt(req.getContentType().length());
            out.writeBytes(req.getContentType().getBytes());
        } else {
            out.writeInt(0);// Length 0
        }

        out.writeInt(-1); // Start of List
        for(ModelInputs input: req.getModelInputs()) {
            encodeModelInputs(input, out);
        }
        out.writeInt(-2); // End of List
    }

    private void encodeModelInputs(ModelInputs modelInputs, ByteBuf out) {
        out.writeInt(modelInputs.getName().length());
        out.writeBytes(modelInputs.getName().getBytes());

        if(modelInputs.getContentType() != null) {
            out.writeInt(modelInputs.getContentType().length());
            out.writeBytes(modelInputs.getContentType().getBytes());
        } else {
            out.writeInt(0); // Length 0
        }

        out.writeInt(modelInputs.getValue().length);
        out.writeBytes(modelInputs.getValue());
    }

    @Override
    protected void encode(ChannelHandlerContext ctx, BaseModelRequest msg, ByteBuf out) throws Exception {
        long startTime = System.nanoTime();
        if (msg instanceof ModelLoadModelRequest) {
            out.writeDouble(1.0); // SOM
            out.writeInt(1); // load 1
            out.writeInt(msg.getModelName().length());
            out.writeBytes(msg.getModelName().getBytes());

            out.writeInt(((ModelLoadModelRequest) msg).getModelPath().length());
            out.writeBytes(((ModelLoadModelRequest) msg).getModelPath().getBytes());

            if (((ModelLoadModelRequest) msg).getBatchSize() >= 0) {
                out.writeInt(((ModelLoadModelRequest) msg).getBatchSize());
            } else {
                out.writeInt(1);
            }

            out.writeInt(((ModelLoadModelRequest) msg).getHandler().length());
            out.writeBytes(((ModelLoadModelRequest) msg).getHandler().getBytes());

            if (((ModelLoadModelRequest) msg).getGpu() != null) {
                out.writeInt(Integer.getInteger(((ModelLoadModelRequest) msg).getGpu()));
            } else {
                out.writeInt(-1);
            }
        } else if (msg instanceof ModelInferenceRequest) {
            out.writeDouble(1.0);
            out.writeInt(2); // Predict/inference: 2
            out.writeInt(msg.getModelName().length());
            out.writeBytes(msg.getModelName().getBytes());

            out.writeInt(-1); // Start of List
            for (RequestBatch batch : ((ModelInferenceRequest) msg).getRequestBatch()) {
                encodeRequestBatch(batch, out);
            }
            out.writeInt(-2); // End of List
        }
        out.writeBytes("\r\n".getBytes()); // EOM
        long endTime = System.nanoTime();
        logger.info("Encode took " + (endTime - startTime) + " ns");
    }

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) throws Exception {
        if (in.getByte(in.readableBytes() - 1) == '\n') {
            long startTime = System.nanoTime();

            ModelWorkerResponse response = new ModelWorkerResponse();

            Double version = in.readDouble();
            response.setCode(Integer.toString(in.readInt()));
            Integer length = in.readInt();
            response.setMessage(in.readCharSequence(length, StandardCharsets.UTF_8).toString());
            length = in.readInt();
            List<Predictions> predictionsList = new ArrayList<>();
            if (length < 0) {
                // There are a list of predictions
                while (length != -2) {
                    Predictions p = new Predictions();
                    length = in.readInt();
                    if (length < 0) continue;
                    p.setRequestId(in.readCharSequence(length, StandardCharsets.UTF_8).toString());
                    Integer code = in.readInt();

                    length = in.readInt();
                    if (length < 0) continue;
                    String encoding = in.readCharSequence(length, StandardCharsets.UTF_8).toString();

                    length = in.readInt();
                    if (length < 0) continue;
                    p.setResp(new byte[length]);
                    if ((encoding.equalsIgnoreCase("json")) ||
                            (encoding.equalsIgnoreCase("text"))) {
                        in.readBytes(p.getResp(), 0, length);
                    } else {
                        // Assuming its binary
                        in.readBytes(p.getResp(), 0, length);
                    }
                    predictionsList.add(p);
                }
                response.setPredictions(predictionsList);
            }

            // read the delimiter bytes
            in.readBytes(in.readableBytes());
            out.add(response);
            logger.info("Decode took " + (System.nanoTime() - startTime) + " ns");
        }
    }
}
