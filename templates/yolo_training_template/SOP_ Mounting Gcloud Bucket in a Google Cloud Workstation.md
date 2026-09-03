# SOP: Mounting Gcloud Bucket in a Google Cloud Workstation

## Step 1: Launch your Google Cloud Workstation

* Open your web browser and navigate to the cloud workstation page: [https://console.cloud.google.com/workstations/](https://console.cloud.google.com/workstations/)   
* Click on **Launch** on the workstation you want to mount the cloud bucket in

## Step 2: Open the terminal in your workstation

* Start a session in your workstation  
* Open a terminal in your chosen IDE (**Terminal** → **New Terminal** or Ctrl \+ \~ in VSCode)

## Step 3: Authenticate in Google Cloud 

```shell
gcloud auth application-default login --no-launch-browser
```

## Step 4: Install `gcsfuse` 

```shell
# Add repository source
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/gcsfuse.gpg
```

```shell
echo "deb [signed-by=/etc/apt/trusted.gpg.d/gcsfuse.gpg] https://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
```

```shell
sudo apt-get update
sudo apt-get install -y gcsfuse
```

## Step 5: Create Mount Directory & Connect Bucket 

* Replace **BUCKET NAME** with the name of your bucket

```shell
# Create target directory
mkdir -p ~/my_gcs_bucket
# Mount GCS bucket 
gcsfuse --implicit-dirs BUCKET NAME ~/my_gcs_bucket

# ex:
# gcsfuse --implicit-dirs nmfs-dev-uc1-pifsc ~/my_gcs_bucket 
```

## Step 6: Verify Connection

```shell
ls -la ~/my_gcs_bucket 
```

## 

## How to re-mount after first set up

* Replace **BUCKET NAME** with the name of your bucket

```shell
gcsfuse --implicit-dirs BUCKET NAME ~/my_gcs_bucket
```

## How to un-mount 

```shell
fusermount -u ~/my_gcs_bucket 
```

