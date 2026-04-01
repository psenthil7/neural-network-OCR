/**
 * Copyright (c) 2025 Sami Menik, PhD. All rights reserved.
 *
 * Unauthorized copying of this file, via any medium, is strictly prohibited.
 * This software is provided "as is," without warranty of any kind.
 */
package cs2725.web;

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import cs2725.api.nn.NeuralNetwork;
import cs2725.impl.nn.SimpleNeuralNetwork;

/**
 * Minimal HTTP API for digit prediction. Intended for deployment behind a reverse proxy
 * (e.g. Railway, Fly.io, Render). Vercel serves the static UI and forwards requests via
 * {@code api/predict.js} using the {@code JAVA_PREDICT_URL} environment variable.
 */
public final class DigitHttpServer {

    private DigitHttpServer() {}

    public static void main(String[] args) throws IOException {
        String weightsPath = System.getenv().getOrDefault(
                "WEIGHTS_PATH",
                "resources" + File.separator + "digits_network_weights.txt");
        NeuralNetwork network = new SimpleNeuralNetwork(weightsPath);

        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8080"));
        HttpServer http = HttpServer.create(new InetSocketAddress(port), 0);

        http.createContext("/health", DigitHttpServer::handleHealth);
        http.createContext("/predict", ex -> handlePredict(ex, network));

        http.setExecutor(null);
        http.start();
        System.err.println("DigitHttpServer listening on port " + port);
    }

    private static void handleHealth(HttpExchange exchange) throws IOException {
        if ("GET".equals(exchange.getRequestMethod())) {
            byte[] ok = "{\"status\":\"ok\"}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
            exchange.sendResponseHeaders(200, ok.length);
            exchange.getResponseBody().write(ok);
        } else {
            exchange.sendResponseHeaders(405, -1);
        }
        exchange.close();
    }

    private static void handlePredict(HttpExchange exchange, NeuralNetwork network) throws IOException {
        addCors(exchange);
        if ("OPTIONS".equals(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            return;
        }
        if (!"POST".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, "{\"error\":\"method not allowed\"}");
            return;
        }
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        float[] input;
        try {
            input = parseInputArray(body);
        } catch (IllegalArgumentException e) {
            sendJson(exchange, 400, "{\"error\":\"expected JSON body {\\\"input\\\":[784 floats]}\"}");
            return;
        }
        float[] out = network.predict(input);
        int digit = network.toLabel(out);
        String json = buildResponseJson(digit, out);
        sendJson(exchange, 200, json);
    }

    private static void addCors(HttpExchange exchange) {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        addCors(exchange);
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    /**
     * Parses {@code {"input":[...]}} with exactly 784 comma-separated floats.
     */
    static float[] parseInputArray(String body) {
        int key = body.indexOf("\"input\"");
        if (key < 0) {
            throw new IllegalArgumentException("missing input");
        }
        int open = body.indexOf('[', key);
        if (open < 0) {
            throw new IllegalArgumentException("missing [");
        }
        int depth = 0;
        int end = open;
        for (; end < body.length(); end++) {
            char c = body.charAt(end);
            if (c == '[') {
                depth++;
            } else if (c == ']') {
                depth--;
                if (depth == 0) {
                    break;
                }
            }
        }
        if (depth != 0) {
            throw new IllegalArgumentException("unbalanced brackets");
        }
        String inner = body.substring(open + 1, end).trim();
        if (inner.isEmpty()) {
            throw new IllegalArgumentException("empty array");
        }
        String[] parts = inner.split(",");
        if (parts.length != 784) {
            throw new IllegalArgumentException("need 784 values, got " + parts.length);
        }
        float[] f = new float[784];
        for (int i = 0; i < 784; i++) {
            f[i] = Float.parseFloat(parts[i].trim());
        }
        return f;
    }

    static String buildResponseJson(int digit, float[] scores) {
        StringBuilder sb = new StringBuilder(512);
        sb.append("{\"digit\":").append(digit).append(",\"scores\":[");
        for (int i = 0; i < scores.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(scores[i]);
        }
        sb.append("]}");
        return sb.toString();
    }
}
