#  **Open Windows PowerShell and run:**
```
ssh-keygen -t rsa -b 4096 -C "your_email@gmail.com"
```

You will see:
```
Enter file in which to save the key (C:\Users\YOU/.ssh/id_rsa):
```

Just press ENTER.
Then it asks:
```
Enter passphrase (leave empty)
```

Just press ENTER twice.

Now you will have:

Private key → C:\Users\YOURNAME\.ssh\id_rsa

Public key → C:\Users\YOURNAME\.ssh\id_rsa.pub

# **Copy PUBLIC KEY**
Run this command:
```
cat ~/.ssh/id_rsa.pub
```

You will get something like:
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQ123.... user@DESKTOP-xxxx
```

Copy the entire line.

# **Paste into CheapCuda**