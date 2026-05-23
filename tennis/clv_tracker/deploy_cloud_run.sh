#!/bin/bash

# Configuración del proyecto y variables
PROJECT_ID="project-5c14344c-37b1-41f2-ade"
REGION="us-central1"
SERVICE_NAME="clv-tracker"
QUEUE_NAME="clv-tasks-queue"

echo "=== Configurando proyecto de Google Cloud ==="
gcloud config set project $PROJECT_ID

echo "=== Habilitando servicios necesarios de GCP ==="
gcloud services enable run.googleapis.com \
                       cloudbuild.googleapis.com \
                       cloudtasks.googleapis.com \
                       artifactregistry.googleapis.com

echo "=== Desplegando en Google Cloud Run ==="
# --source . se encarga de subir el código a Cloud Build, construir la imagen
# usando el Dockerfile local, registrarla en Artifact Registry y desplegar en Cloud Run.
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --allow-unauthenticated \
    --update-env-vars="SUPABASE_URL=https://ncarrwurngrewhiigcqf.supabase.co,GCP_PROJECT=$PROJECT_ID,GCP_LOCATION=$REGION,QUEUE_NAME=$QUEUE_NAME"

echo "=== Despliegue completado ==="
echo "Por favor, recuerda configurar las variables de entorno restantes (SUPABASE_KEY, ODDS_API_KEY, SERVICE_URL, API_SECRET) en la consola de Cloud Run."
