{{/*
Expand the name of the chart.
*/}}
{{- define "trace.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "trace.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "trace.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "trace.labels" -}}
helm.sh/chart: {{ include "trace.chart" . }}
{{ include "trace.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: qbrix
app.kubernetes.io/component: trace
{{- end }}

{{/*
Selector labels
*/}}
{{- define "trace.selectorLabels" -}}
app.kubernetes.io/name: {{ include "trace.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "trace.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "trace.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the image repository with optional global registry
*/}}
{{- define "trace.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .Values.image.repository }}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Get redis secret name
*/}}
{{- define "trace.redisSecretName" -}}
{{- if .Values.global.redis.existingSecret }}
{{- .Values.global.redis.existingSecret }}
{{- else }}
{{- printf "%s-redis" .Release.Name }}
{{- end }}
{{- end }}

{{/*
Get clickhouse secret name
*/}}
{{- define "trace.clickhouseSecretName" -}}
{{- if .Values.global.clickhouse.existingSecret }}
{{- .Values.global.clickhouse.existingSecret }}
{{- else }}
{{- printf "%s-clickhouse" .Release.Name }}
{{- end }}
{{- end }}
