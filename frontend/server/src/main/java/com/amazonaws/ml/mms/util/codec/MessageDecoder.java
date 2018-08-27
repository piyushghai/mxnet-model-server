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

import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;


public class MessageDecoder extends ByteToMessageDecoder {
    private final Logger logger = LoggerFactory.getLogger(MessageDecoder.class);
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
