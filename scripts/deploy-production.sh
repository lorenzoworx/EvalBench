#!/usr/bin/env bash
set -Eeuo pipefail

: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
: "${DEPLOY_STACK_DIR:?DEPLOY_STACK_DIR is required}"
: "${PRODUCTION_URL:?PRODUCTION_URL is required}"

short_sha=${GITHUB_SHA:0:12}
release_image="personal-web-evalbench:release-${short_sha}"
production_image="personal-web-evalbench:1"
rollback_image="personal-web-evalbench:rollback-before-${short_sha}"
candidate_container="evalbench-candidate-${GITHUB_RUN_ID}"
compose_file="${DEPLOY_STACK_DIR}/compose.yml"
promoted=0

cleanup() {
  status=$?
  set +e
  docker container rm --force "$candidate_container" >/dev/null 2>&1
  if [[ $status -ne 0 && $promoted -eq 1 ]]; then
    echo "Deployment failed after promotion; restoring ${rollback_image}." >&2
    docker image tag "$rollback_image" "$production_image"
    docker compose -f "$compose_file" up -d --no-deps --force-recreate --wait evalbench
  fi
  exit "$status"
}
trap cleanup EXIT

test -f "$compose_file"
docker image inspect "$production_image" >/dev/null
docker image tag "$production_image" "$rollback_image"

docker build \
  --file Dockerfile.demo \
  --label "org.opencontainers.image.revision=${GITHUB_SHA}" \
  --tag "$release_image" \
  .

docker run --detach \
  --name "$candidate_container" \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m,uid=10001,gid=10001 \
  --user 10001:10001 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  "$release_image" >/dev/null

for attempt in {1..30}; do
  if docker exec "$candidate_container" wget -qO- \
      http://127.0.0.1:8000/replay-data.json >/dev/null; then
    break
  fi
  if [[ $attempt -eq 30 ]]; then
    docker logs "$candidate_container"
    exit 1
  fi
  sleep 1
done
candidate_post_response=$(docker exec "$candidate_container" wget --server-response \
  --post-data='' -O /dev/null http://127.0.0.1:8000/api/runs 2>&1 || true)
candidate_post_status=$(sed -n 's/.*HTTP\/1\.[01] \([0-9][0-9][0-9]\).*/\1/p' \
  <<< "$candidate_post_response" | tail -1)
test "$candidate_post_status" = 403

docker container rm --force "$candidate_container" >/dev/null
docker image tag "$release_image" "$production_image"
promoted=1
docker compose -f "$compose_file" up -d --no-deps --force-recreate --wait evalbench

deployed_revision=$(docker inspect personal-web-evalbench-1 --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')
test "$deployed_revision" = "$GITHUB_SHA"
curl --fail --silent --show-error --retry 5 --retry-all-errors \
  "${PRODUCTION_URL%/}/projects/evalbench/replay-data.json" >/dev/null
test "$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --request POST "${PRODUCTION_URL%/}/projects/evalbench/api/runs")" = 403

promoted=0
echo "EvalBench ${GITHUB_SHA} is live; rollback image: ${rollback_image}."
