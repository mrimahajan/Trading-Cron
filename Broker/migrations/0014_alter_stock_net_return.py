# Generated by Django 5.0.2 on 2024-02-25 14:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Broker', '0013_alter_stock_company_alter_stock_eod_price_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stock',
            name='net_return',
            field=models.FloatField(null=True),
        ),
    ]
