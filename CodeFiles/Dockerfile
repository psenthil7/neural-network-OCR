# Use this when Railway "Root Directory" is set to CodeFiles (build context = this folder).
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app
COPY pom.xml .
COPY src ./src
COPY resources ./resources
RUN mvn -q -DskipTests package

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY --from=build /app/target/digit-server.jar /app/digit-server.jar
COPY resources /app/resources
ENV PORT=8080
EXPOSE 8080
CMD ["java", "-jar", "/app/digit-server.jar"]
