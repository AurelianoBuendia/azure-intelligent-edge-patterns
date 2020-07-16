# Generated by Django 3.0.7 on 2020-07-15 08:17

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Part',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('description', models.CharField(blank=True, default='', max_length=1000)),
                ('is_demo', models.BooleanField(default=False)),
                ('name_lower', models.CharField(default='<django.db.models.fields.charfield>', max_length=200)),
            ],
            options={
                'unique_together': {('name_lower', 'is_demo')},
            },
        ),
    ]
