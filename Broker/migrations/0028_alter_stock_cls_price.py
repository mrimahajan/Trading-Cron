# Generated by Django 5.0.2 on 2024-02-26 03:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Broker', '0027_alter_stock_cls_price'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stock',
            name='CLS_Price',
            field=models.FloatField(default=0, null=True),
        ),
    ]
