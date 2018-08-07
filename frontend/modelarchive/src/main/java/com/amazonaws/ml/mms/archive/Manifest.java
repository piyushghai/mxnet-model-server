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
package com.amazonaws.ml.mms.archive;

import com.google.gson.annotations.SerializedName;

public class Manifest {

    private String specificationVersion;
    private String implementationVersion;
    private String description;
    private String modelServerVersion;
    private String license;
    private Engine engine;
    private Model model;
    private Publisher publisher;

    public Manifest() {
        specificationVersion = "1.0";
        implementationVersion = "1.0";
        modelServerVersion = "1.0";
        license = "Apache 2.0";
        engine = new Engine();
    }

    public String getSpecificationVersion() {
        return specificationVersion;
    }

    public void setSpecificationVersion(String specificationVersion) {
        this.specificationVersion = specificationVersion;
    }

    public String getImplementationVersion() {
        return implementationVersion;
    }

    public void setImplementationVersion(String implementationVersion) {
        this.implementationVersion = implementationVersion;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getModelServerVersion() {
        return modelServerVersion;
    }

    public void setModelServerVersion(String modelServerVersion) {
        this.modelServerVersion = modelServerVersion;
    }

    public String getLicense() {
        return license;
    }

    public void setLicense(String license) {
        this.license = license;
    }

    public Engine getEngine() {
        return engine;
    }

    public void setEngine(Engine engine) {
        this.engine = engine;
    }

    public Model getModel() {
        return model;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    public Publisher getPublisher() {
        return publisher;
    }

    public void setPublisher(Publisher publisher) {
        this.publisher = publisher;
    }

    public static final class Publisher {

        private String author;
        private String email;

        public Publisher() {}

        public String getAuthor() {
            return author;
        }

        public void setAuthor(String author) {
            this.author = author;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }
    }

    public static final class Engine {

        private EngineType engineName;
        private String engineVersion;
        private RuntimeType runtime;

        public Engine() {
            engineName = EngineType.NONE;
            runtime = RuntimeType.PYTHON2_7;
        }

        public EngineType getEngineName() {
            return engineName;
        }

        public void setEngineName(EngineType engineName) {
            this.engineName = engineName;
        }

        public String getEngineVersion() {
            return engineVersion;
        }

        public void setEngineVersion(String engineVersion) {
            this.engineVersion = engineVersion;
        }

        public RuntimeType getRuntime() {
            return runtime;
        }

        public void setRuntime(RuntimeType runtime) {
            this.runtime = runtime;
        }
    }

    public static final class Model {

        private String modelName;
        private String description;
        private String modelVersion;
        private String parametersFile;
        private String symbolFile;
        private String handler;

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public String getModelVersion() {
            return modelVersion;
        }

        public void setModelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
        }

        public String getParametersFile() {
            return parametersFile;
        }

        public void setParametersFile(String parametersFile) {
            this.parametersFile = parametersFile;
        }

        public String getSymbolFile() {
            return symbolFile;
        }

        public void setSymbolFile(String symbolFile) {
            this.symbolFile = symbolFile;
        }

        public String getHandler() {
            return handler;
        }

        public void setHandler(String handler) {
            this.handler = handler;
        }
    }

    public enum EngineType {
        @SerializedName("MxNet")
        MX_NET("MxNet"),
        @SerializedName("None")
        NONE("None");

        String value;

        EngineType(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        public static EngineType fromValue(String value) {
            for (EngineType engineType : values()) {
                if (engineType.value.equals(value)) {
                    return engineType;
                }
            }
            throw new IllegalArgumentException("Invalid EngineType value: " + value);
        }
    }

    public enum RuntimeType {
        @SerializedName("python2.7")
        PYTHON2_7("python2.7"),
        @SerializedName("python3.6")
        PYTHON3_6("python3.6");

        String value;

        RuntimeType(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        public static RuntimeType fromValue(String value) {
            for (RuntimeType runtime : values()) {
                if (runtime.value.equals(value)) {
                    return runtime;
                }
            }
            throw new IllegalArgumentException("Invalid RuntimeType value: " + value);
        }
    }
}