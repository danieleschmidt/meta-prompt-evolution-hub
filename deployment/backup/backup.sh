#!/bin/bash
# Database backup script for Meta-Prompt-Evolution-Hub
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="${DB_NAME:-metapromptevolution}"

echo "🗄️  Starting database backup: $TIMESTAMP"

# Create backup directory
mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump $DB_NAME | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Upload to S3 (if configured)
if [ ! -z "${S3_BACKUP_BUCKET:-}" ]; then
    aws s3 cp "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz" "s3://$S3_BACKUP_BUCKET/db_backups/"
    echo "✅ Backup uploaded to S3"
fi

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

echo "✅ Database backup completed: $TIMESTAMP"
