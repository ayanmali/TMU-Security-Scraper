# Generated by Django 5.1.3 on 2024-11-16 13:44

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_alter_incident_options'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='incident',
            table='incidents',
        ),
    ]
